"""how much of the fp64 interaction error comes from transformers' fp32 softmax cast, with and without the bias?

transformers' gpt-neox attention asks softmax for dtype=float32, so an "fp64" model still rounds every
attention score to fp32 before softmax. baligar (2026) found that this cast dominates interaction
measurement in nominal fp64. this runs the same 560 pairs as scripts/patching_floor.py in fp64 with the
stock softmax (rotary kept in fp64, so the softmax is the only change) and compares against the true
fp64 runs on disk, as released and with the invariant key bias dropped. it also records the largest
attention score offset on the prompts, from one fp32 pass.

usage: python scripts/fp64_softmax_cast.py
writes evidence/patching_floor/fp64_softmax_cast.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.models.precision import drop_invariant_key_bias

sys.path.insert(0, str(Path(__file__).parent))
from bf16_audit import pick_prompts  # noqa: E402
from patching_floor import OUT, fp64_rotary, interactions  # noqa: E402
from score_offsets import Catcher  # noqa: E402

MODELS = ("pythia-410m-deduped", "pythia-410m")


@torch.no_grad()
def offsets(adapter, cases):
    cat = Catcher(1.0)
    real = F.softmax

    def spy(x, *a, **k):
        if x.dim() == 4:
            cat.see(x)
        return real(x, *a, **k)

    F.softmax = spy
    try:
        for c in cases:
            cat.layer = 0
            adapter.model(input_ids=torch.tensor([c.corrupt_ids], device=adapter.device))
    finally:
        F.softmax = real
    s = cat.summary()
    return {k: s[k] for k in ("largest_offset", "largest_offset_layer", "spread_there")}


def main():
    result = {}
    for name in MODELS:
        ref = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        model_id, rev = ref["model"], ref["revision"]
        configure_determinism(0, True)
        tok = load_tokenizer(model_id, rev)
        fp32 = load(model_id, rev, "eager", "float32", "cuda")
        cases = pick_prompts(tok, fp32, ref["n_prompts"])
        if [c.case.corrupt for c in cases] != ref["prompts"]:
            raise RuntimeError(f"{name}: prompts differ from the floor run")
        out = {"offsets_fp32": offsets(fp32, cases)}
        del fp32
        torch.cuda.empty_cache()
        for drop in (False, True):
            a = load(model_id, rev, "eager", "float64", "cuda")
            if drop:
                drop_invariant_key_bias(a)
            fp64_rotary(a)  # stock softmax, fp64 rotary: the softmax cast is the only difference
            rows = [interactions(a, c) for c in cases]
            key = "float64, bias dropped" if drop else "float64"
            q = np.array([r["q"] for r in rows]).ravel()
            q64 = np.array([r["q"] for r in ref["runs"][key]["rows"]]).ravel()
            q32 = np.array([r["q"] for r in ref["runs"]["float32" + (", bias dropped" if drop else "")]["rows"]]).ravel()
            label = "bias dropped" if drop else "as released"
            out[label] = {"cast_error_median": float(np.median(np.abs(q - q64))),
                          "fp32_floor": float(np.median(np.abs(q32 - q64)))}
            del a
            torch.cuda.empty_cache()
            print(f"{name:20s} {label:13s} fp32-softmax-cast error {out[label]['cast_error_median']:.2e}   "
                  f"full fp32 floor {out[label]['fp32_floor']:.2e}", flush=True)
        print(f"{name:20s} largest offset {out['offsets_fp32']['largest_offset']:,.0f} "
              f"(layer {out['offsets_fp32']['largest_offset_layer']}, spread {out['offsets_fp32']['spread_there']:.1f})")
        result[name] = out
    (OUT / "fp64_softmax_cast.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print("wrote", OUT / "fp64_softmax_cast.json")


if __name__ == "__main__":
    main()
