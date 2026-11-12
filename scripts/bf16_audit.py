"""how much does bf16 move an activation patching result, and does an fp32 head fix it?

for each model and each of four settings (fp32, bf16, bf16 with only the
unembedding in fp32, bf16 with attention and the unembedding in fp32) this measures, on ioi prompts the model actually solves:

- the batch-shape jump: the corrupted prompt's logit diff run alone versus as
  row 0 of a batch of identical copies. the math is the same, so in exact
  arithmetic the gap is zero.
- per-layer mlp restoration, patching the clean mlp output of one layer into
  the corrupted run. once with each layer in its own batch of one (the same
  shape as the baseline), and once with the whole layer sweep stacked into one
  batch, row l patching layer l, which is the usual way to make a sweep fast.

fp32 is the reference. everything else is reported as a distance from it.

usage: python scripts/bf16_audit.py [n_prompts] [model ...] [--task capitals]
writes evidence/bf16_audit/<model>.json, or evidence/bf16_audit_capitals/<model>.json
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kerneltrace.core.manifest import configure_determinism, environment
from kerneltrace.interventions.patching import Patch, margins, run
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.models.precision import drop_invariant_key_bias, upcast_attention
from kerneltrace.tasks import capitals, ioi
from kerneltrace.tasks.preflight import behavioural_check, single_token, structural_check

MODELS = {
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "pythia-160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"),
    "pythia-410m": ("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
    "pythia-1b": ("EleutherAI/pythia-1b", "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"),
}
# label, dtype, fp32 head, fp32 attention, invariant key bias dropped
SETTINGS = (("fp32", "float32", False, False, False), ("bf16", "bfloat16", False, False, False),
            ("bf16 + fp32 head", "bfloat16", True, False, False),
            ("bf16 + fp32 attention and head", "bfloat16", True, True, False),
            ("bf16, key bias dropped", "bfloat16", False, False, True),
            ("bf16, key bias dropped + fp32 head", "bfloat16", True, False, True))
OUT = Path("evidence/bf16_audit")
OUTS = {"ioi": OUT, "capitals": Path("evidence/bf16_audit_capitals")}


def bf16_step(x: float) -> float:
    """distance between neighbouring bf16 values near x (8 bits of significand)."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0.0


def pick_prompts(tok, adapter, n, task="ioi"):
    """task pairs that tokenize cleanly and that the fp32 model actually gets right."""
    if task == "capitals":
        pool = capitals.generate(3000)
    else:
        names_ok = lambda name: single_token(tok, " " + name) is not None
        pool = ioi.generate("discovery", 200, 7, names_ok) + ioi.generate("holdout", 200, 7, names_ok)
    keep = []
    for raw in pool:
        case = structural_check(raw, tok, 48)[0]
        if case is None:
            continue
        mc, mx = one_margin(adapter, case.clean_ids, case), one_margin(adapter, case.corrupt_ids, case)
        if behavioural_check(mc, mx, 1.0, True):
            continue
        keep.append(case)
        if len(keep) == n:
            break
    return keep


def ids(adapter, case, which):
    return torch.tensor([getattr(case, which)], device=adapter.device)


def ab(adapter, case, rows=1):
    dev = adapter.device
    return (torch.full((rows,), case.answer_id, device=dev), torch.full((rows,), case.distractor_id, device=dev))


def one_margin(adapter, token_ids, case, fp32_logits=False):
    x = torch.tensor([token_ids], device=adapter.device)
    a, b = ab(adapter, case)
    return float(margins(run(adapter, x, fp32_logits=fp32_logits)[0], a, b)[0])


@torch.no_grad()
def measure(adapter, case, fp32_logits):
    sites = [("mlp_out", layer) for layer in range(adapter.n_layers)]
    clean, corrupt = ids(adapter, case, "clean_ids"), ids(adapter, case, "corrupt_ids")
    a, b = ab(adapter, case)
    logits_c, cache = run(adapter, clean, capture=sites, fp32_logits=fp32_logits)
    logits_x, _ = run(adapter, corrupt, fp32_logits=fp32_logits)
    m_clean = float(margins(logits_c, a, b)[0])
    m_corrupt = float(margins(logits_x, a, b)[0])
    answer_logit = float(logits_x[0, -1, case.answer_id])

    # the same corrupted prompt as row 0 of a batch of T copies
    T = corrupt.shape[1]
    aT, bT = ab(adapter, case, T)
    m_batched = float(margins(run(adapter, corrupt.repeat(T, 1), fp32_logits=fp32_logits)[0], aT, bT)[0])

    # in bf16 the clean and corrupted margins can round to the same value, and then
    # there is no effect left to restore. that is a result, so it is kept as nan.
    contrast = (m_clean - m_corrupt) or float("nan")
    same = []
    for site in sites:
        logits, _ = run(adapter, corrupt, patches={site: Patch(cache[site])}, fp32_logits=fp32_logits)
        same.append((float(margins(logits, a, b)[0]) - m_corrupt) / contrast)

    # the whole sweep as one batch: row l gets the clean mlp output at layer l only
    L = adapter.n_layers
    hooks = []
    for layer, site in enumerate(sites):
        def patch(_m, _i, out, layer=layer, src=cache[site]):
            out = out.clone()
            out[layer] = src[0].to(out.dtype)
            return out
        hooks.append(adapter.mlp(layer).register_forward_hook(patch))
    try:
        aL, bL = ab(adapter, case, L)
        logits = run(adapter, corrupt.repeat(L, 1), fp32_logits=fp32_logits)[0]
        sweep = ((margins(logits, aL, bL).cpu().numpy() - m_corrupt) / contrast).tolist()
    finally:
        for h in hooks:
            h.remove()

    return {"clean": m_clean, "corrupt": m_corrupt, "corrupt_batched": m_batched,
            "answer_logit": answer_logit, "restore_same_shape": same, "restore_sweep_batched": sweep}


def audit(name, n, task="ioi"):
    model_id, rev = MODELS[name]
    configure_determinism(0, True)
    tok = load_tokenizer(model_id, rev)
    out = OUTS[task]
    path = out / f"{name}.json"
    # settings already on disk are kept, so adding a setting only runs the new one
    result = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
              else {"model": model_id, "revision": rev, "settings": {}, "prompts": []})
    cases = None
    for label, dtype, fp32_logits, fp32_attention, drop_bias in SETTINGS:
        if label in result["settings"] and len(result["settings"][label]["rows"]) == n:
            continue
        if cases is None:  # chosen on the fp32 reference, then held fixed
            cases = pick_prompts(tok, load(model_id, rev, "eager", "float32", "cuda"), n, task)
            torch.cuda.empty_cache()
            if result["prompts"] and [c.case.case_id for c in cases] != [p["case_id"] for p in result["prompts"]]:
                raise RuntimeError(f"{name}: picked different prompts than the ones on disk")
        adapter = load(model_id, rev, "eager", dtype, "cuda")
        if fp32_attention:
            upcast_attention(adapter)
        if drop_bias:
            drop_invariant_key_bias(adapter)
        if not result["prompts"]:
            result["prompts"] = [{"case_id": c.case.case_id, "corrupt": c.case.corrupt,
                                  "answer": c.case.answer.strip(), "distractor": c.case.distractor.strip()}
                                 for c in cases]
            result["environment"] = {k: v for k, v in environment().items() if k != "packages"}
        t0 = time.time()
        rows = [measure(adapter, c, fp32_logits) for c in cases]
        result["settings"][label] = {"dtype": dtype, "fp32_logits": fp32_logits,
                                     "fp32_attention": fp32_attention,
                                     "drop_key_bias": drop_bias, "rows": rows,
                                     "seconds": round(time.time() - t0, 1)}
        gaps = [abs(r["corrupt"] - r["corrupt_batched"]) for r in rows]
        print(f"{name:12s} {label:34s} prompts {len(rows)}  batch gap median {np.median(gaps):.4f} "
              f"max {max(gaps):.4f}  answer logit ~{np.median([r['answer_logit'] for r in rows]):.0f}", flush=True)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8", newline="\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    task = "ioi"
    if "--task" in args:
        i = args.index("--task")
        task = args[i + 1]
        del args[i:i + 2]
    n = int(args[0]) if args else 48
    for name in args[1:] or list(MODELS):
        audit(name, n, task)
