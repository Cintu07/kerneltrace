"""Causal traces for GPT-2 small on IOI: restore one (token, layer) site at a time.

For each valid prompt pair and each site kind (residual stream after the block,
MLP output, attention output), the clean activation at one token position and
one layer is patched into the corrupted run. The cell value is the fraction of
the clean-minus-corrupt logit contrast that comes back. Done in fp32 (the
reference) and bf16 (to show what reduced precision does to the picture).

Usage: python scripts/causal_trace.py [n_per_split]
Writes evidence/causal_trace/gpt2_ioi_trace.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kerneltrace.core.manifest import configure_determinism, environment
from kerneltrace.interventions.patching import margins, run
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.tasks import ioi
from kerneltrace.tasks.preflight import behavioural_check, single_token, structural_check

MODEL, REV = "openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
KINDS = ("resid", "mlp", "attn")


def module(adapter, kind, layer):
    block = adapter.blocks()[layer]
    return {"resid": block, "mlp": block.mlp, "attn": block.attn}[kind]


def first(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def trace(adapter, case):
    """[kind][layer][pos] restoration for one prompt pair."""
    dev = adapter.device
    clean = torch.tensor([case.clean_ids], device=dev)
    corrupt = torch.tensor([case.corrupt_ids], device=dev)
    a, b = torch.tensor([case.answer_id], device=dev), torch.tensor([case.distractor_id], device=dev)
    T = clean.shape[1]
    cache = {}
    hooks = []
    for kind in KINDS:
        for layer in range(adapter.n_layers):
            def save(_m, _i, out, key=(kind, layer)):
                cache[key] = first(out).detach().clone()
            hooks.append(module(adapter, kind, layer).register_forward_hook(save))
    m_clean = float(margins(adapter.model(input_ids=clean).logits, a, b)[0])
    for h in hooks:
        h.remove()
    m_corrupt = float(margins(run(adapter, corrupt)[0], a, b)[0])
    batch = corrupt.repeat(T, 1)              # row p patches position p
    rows = torch.arange(T, device=dev)
    out = {}
    for kind in KINDS:
        grid = []
        for layer in range(adapter.n_layers):
            src = cache[(kind, layer)]

            def patch(_m, _i, o, src=src):
                h = first(o).clone()
                h[rows, rows] = src[0, rows].to(h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

            hk = module(adapter, kind, layer).register_forward_hook(patch)
            try:
                pm = margins(adapter.model(input_ids=batch).logits, a.repeat(T), b.repeat(T)).cpu().numpy()
            finally:
                hk.remove()
            contrast = m_clean - m_corrupt
            grid.append(((pm - m_corrupt) / contrast if contrast != 0 else np.full(T, np.nan)).tolist())
        out[kind] = grid
    return out, m_clean, m_corrupt


def roles(case, tok):
    ids = case.clean_ids
    io_id = single_token(tok, case.case.answer)
    s_id = single_token(tok, case.case.distractor)
    s2 = next(i for i, (x, y) in enumerate(zip(case.clean_ids, case.corrupt_ids)) if x != y)
    io = ids.index(io_id)
    s1 = ids.index(s_id)
    return {"io": io, "s1": s1, "s2": s2, "end": len(ids) - 1}


def main(n_per_split: int = 24):
    configure_determinism(0, True)
    tok = load_tokenizer(MODEL, REV)
    names_ok = lambda n: single_token(tok, " " + n) is not None
    candidates = ioi.generate("discovery", 80, 7, names_ok) + ioi.generate("holdout", 80, 7, names_ok)
    structural = [c for c in (structural_check(x, tok, 48)[0] for x in candidates) if c is not None]
    result = {"model": MODEL, "revision": REV, "kinds": list(KINDS), "environment": None, "prompts": []}
    models = {"float32": load(MODEL, REV, "eager", "float32", "cuda"),
              "bfloat16": load(MODEL, REV, "eager", "bfloat16", "cuda")}
    result["environment"] = {k: v for k, v in environment().items() if k != "packages"}
    t0 = time.time()
    taken = {"discovery": 0, "holdout": 0}
    for case in structural:
        split = case.case.split
        if taken[split] >= n_per_split:
            continue
        tr32, mc, mx = trace(models["float32"], case)
        if behavioural_check(mc, mx, 1.0, True):
            continue  # the model does not do the task on this prompt (fp32 reference)
        tr16, mc16, mx16 = trace(models["bfloat16"], case)
        taken[split] += 1
        result["prompts"].append({
            "case_id": case.case.case_id, "split": split, "template": case.case.template,
            "clean": case.case.clean, "corrupt": case.case.corrupt,
            "answer": case.case.answer.strip(), "distractor": case.case.distractor.strip(),
            "tokens": [tok.decode([t]) for t in case.corrupt_ids],
            "clean_tokens": [tok.decode([t]) for t in case.clean_ids],
            "roles": roles(case, tok),
            "float32": {"clean_margin": mc, "corrupt_margin": mx, "trace": tr32},
            "bfloat16": {"clean_margin": mc16, "corrupt_margin": mx16, "trace": tr16},
        })
        print(f"{len(result['prompts']):3d} {split:9s} {case.case.corrupt[:60]!r}  "
              f"clean {mc:+.2f} corrupt {mx:+.2f}", flush=True)
        if all(v >= n_per_split for v in taken.values()):
            break
    result["seconds"] = round(time.time() - t0, 1)
    out = Path("evidence/causal_trace/gpt2_ioi_trace.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(result['prompts'])} prompts in {result['seconds']}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
