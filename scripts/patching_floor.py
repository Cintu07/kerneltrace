"""does the invariant key bias set the fp32 floor under interaction patching (baligar 2026)?

baligar measured pair interactions q_ij = f11 - f10 - f01 + f00 from residual stream patching in
fp32 and fp64, and found pythia-410m-deduped's floor (median |q32 - q64|) 10x gpt-2-medium's. this
reruns that kind of measurement on ioi, with and without the softmax-invariant key bias, and adds
bf16.

sites: the residual stream entering 4 blocks (L/6, 2L/6, 3L/6, 4L/6) at 2 positions (the flipped
token and the last token). 8 sites, 28 pairs per prompt, full displacement (corrupt to clean).
the fp64 reference keeps softmax and rotary in fp64: transformers casts both to fp32 for gpt-neox.

usage: python scripts/patching_floor.py [n_prompts] [model ...]
writes evidence/patching_floor/<model>.json
"""
from __future__ import annotations

import gc
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.models.precision import drop_invariant_key_bias

sys.path.insert(0, str(Path(__file__).parent))
from bf16_audit import pick_prompts  # noqa: E402

MODELS = {
    "pythia-410m-deduped": ("EleutherAI/pythia-410m-deduped", None),
    "pythia-410m": ("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
    "gpt2-medium": ("openai-community/gpt2-medium", None),
}
PRECISIONS = ("float64", "float32", "bfloat16")
OUT = Path("evidence/patching_floor")


class Fp64Softmax:
    """transformers asks gpt-neox's softmax for dtype=float32. in an fp64 model, keep fp64."""

    def __enter__(self):
        self.real = F.softmax

        def softmax(x, *a, **k):
            if x.dtype == torch.float64 and k.get("dtype") == torch.float32:
                k["dtype"] = torch.float64
            return self.real(x, *a, **k)

        F.softmax = softmax

    def __exit__(self, *exc):
        F.softmax = self.real


def fp64_rotary(adapter):
    """replace gpt-neox's rotary (computed in fp32, then cast) with the same thing in fp64."""
    rot = adapter.model.gpt_neox.rotary_emb
    attn = adapter.attention(0)
    dim = attn.rotary_ndims
    base = getattr(adapter.config, "rotary_emb_base", None) or adapter.config.rope_theta
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64, device=adapter.device) / dim))
    if not torch.allclose(inv.float(), rot.inv_freq.float(), rtol=1e-6):
        raise RuntimeError("rebuilt rotary frequencies do not match the model's")

    def forward(x, position_ids):
        freqs = (inv[None, :, None] @ position_ids[:, None, :].double()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    rot.forward = forward


def sites_for(adapter, case):
    L = adapter.n_layers
    depths = [round(L * k / 6) for k in (1, 2, 3, 4)]
    flip = next(i for i, (a, b) in enumerate(zip(case.clean_ids, case.corrupt_ids)) if a != b)
    return [(d, p) for d in depths for p in (flip, len(case.clean_ids) - 1)]


@torch.no_grad()
def forward(adapter, ids, patches, case):
    """patches: {(layer, pos): tensor [d]} written into the residual stream entering that block."""
    handles = []
    for (layer, pos), vec in patches.items():
        def pre(_m, args, kwargs, pos=pos, vec=vec):
            if args:
                h = args[0].clone(); h[0, pos] = vec
                return (h, *args[1:]), kwargs
            h = kwargs["hidden_states"].clone(); h[0, pos] = vec
            kwargs["hidden_states"] = h
            return args, kwargs
        handles.append(adapter.blocks()[layer].register_forward_pre_hook(pre, with_kwargs=True))
    try:
        logits = adapter.model(input_ids=ids).logits[0, -1].double()
    finally:
        for h in handles:
            h.remove()
    return float(logits[case.answer_id] - logits[case.distractor_id])


@torch.no_grad()
def interactions(adapter, case):
    sites = sites_for(adapter, case)
    clean = torch.tensor([case.clean_ids], device=adapter.device)
    corrupt = torch.tensor([case.corrupt_ids], device=adapter.device)
    store = {}
    handles = [adapter.blocks()[layer].register_forward_pre_hook(
        lambda _m, args, kwargs, layer=layer: store.__setitem__(
            layer, (args[0] if args else kwargs["hidden_states"])[0].clone()), with_kwargs=True)
        for layer in {d for d, _ in sites}]
    try:
        f_clean = float((lambda lg: lg[case.answer_id] - lg[case.distractor_id])(
            adapter.model(input_ids=clean).logits[0, -1].double()))
    finally:
        for h in handles:
            h.remove()
    val = {s: store[s[0]][s[1]] for s in sites}
    f00 = forward(adapter, corrupt, {}, case)
    single = {s: forward(adapter, corrupt, {s: val[s]}, case) for s in sites}
    q, delta = [], [single[s] - f00 for s in sites]
    for a, b in itertools.combinations(sites, 2):
        f11 = forward(adapter, corrupt, {a: val[a], b: val[b]}, case)
        q.append(f11 - single[a] - single[b] + f00)
    return {"clean": f_clean, "corrupt": f00, "delta": delta, "q": q,
            "sites": [list(s) for s in sites]}


def run_model(name, n):
    model_id, rev = MODELS[name]
    if rev is None:
        from huggingface_hub import model_info
        rev = model_info(model_id).sha
    configure_determinism(0, True)
    tok = load_tokenizer(model_id, rev)
    cases = pick_prompts(tok, load(model_id, rev, "eager", "float32", "cuda"), n)
    torch.cuda.empty_cache()
    result = {"model": model_id, "revision": rev, "n_prompts": len(cases),
              "prompts": [c.case.corrupt for c in cases], "runs": {}}
    for drop in (False, True):
        for prec in PRECISIONS:
            adapter = load(model_id, rev, "eager", prec, "cuda")
            removed = drop_invariant_key_bias(adapter) if drop else 0.0
            if prec == "float64" and adapter.family == "gpt_neox":
                fp64_rotary(adapter)
            t0 = time.time()
            with Fp64Softmax():
                rows = [interactions(adapter, c) for c in cases]
            key = f"{prec}{', bias dropped' if drop else ''}"
            result["runs"][key] = {"rows": rows, "removed": removed, "seconds": round(time.time() - t0, 1)}
            print(f"{name:20s} {key:24s} {time.time() - t0:6.1f}s", flush=True)
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result), encoding="utf-8", newline="\n")
    return result


def stats(result):
    runs = result["runs"]
    q = {k: np.array([r["q"] for r in v["rows"]]).ravel() for k, v in runs.items()}
    d = {k: np.array([r["delta"] for r in v["rows"]]).ravel() for k, v in runs.items()}
    out = {"fp64_bias_vs_dropped_max": float(np.abs(q["float64"] - q["float64, bias dropped"]).max()),
           "median_abs_q64": float(np.median(np.abs(q["float64"]))), "pairs": int(q["float64"].size)}
    for drop in ("", ", bias dropped"):
        ref = q["float64" + drop]
        for prec in ("float32", "bfloat16"):
            k = prec + drop
            err = np.abs(q[k] - ref)
            floor = float(np.median(err))
            out[k] = {"floor": floor, "share_below_3x_floor": float(np.mean(np.abs(ref) < 3 * floor)),
                      "sign_disagree": float(np.mean(np.sign(q[k]) != np.sign(ref))),
                      "share_err_over_signal": float(np.mean(err > np.abs(ref))),
                      "single_site_err_median": float(np.median(np.abs(d[k] - d["float64" + drop])))}
    return out


def main(argv):
    n = int(argv[0]) if argv and argv[0].isdigit() else 20
    names = [a for a in argv if not a.isdigit()] or list(MODELS)
    summary = {}
    for name in names:
        res = run_model(name, n)
        s = summary[name] = stats(res)
        print(f"{name}: pairs {s['pairs']}, median |q64| {s['median_abs_q64']:.2e}, fp64 bias vs dropped max "
              f"{s['fp64_bias_vs_dropped_max']:.1e}")
        for k in ("float32", "float32, bias dropped", "bfloat16", "bfloat16, bias dropped"):
            r = s[k]
            print(f"   {k:24s} floor {r['floor']:.2e}  below 3x floor {r['share_below_3x_floor']:6.1%}  "
                  f"sign flips {r['sign_disagree']:6.1%}  err > signal {r['share_err_over_signal']:6.1%}  "
                  f"single-site err {r['single_site_err_median']:.2e}", flush=True)
    path = OUT / "summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(summary)
    path.write_text(json.dumps(old, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1:])
