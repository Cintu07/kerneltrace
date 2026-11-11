"""which part of the model has to stay in fp32? keep one kind of block in fp32, the rest in bf16.

for each choice (nothing, attention, mlp, layernorms, attention and mlp) the corrupted
prompt's logit diff is computed with the unembedding in fp32, so the output grid is out of
the way, and compared with the full fp32 model.

usage: python scripts/precision_bisect.py [model ...]
writes evidence/bf16_audit/precision_bisect.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.interventions.patching import margins, run
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.models.precision import _cast

MODELS = {
    "pythia-160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"),
    "pythia-410m": ("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
}
PARTS = {"nothing": (), "attention": ("attention",), "mlp": ("mlp",),
         "layernorms": ("input_layernorm", "post_attention_layernorm"), "attention and mlp": ("attention", "mlp")}
OUT = Path("evidence/bf16_audit/precision_bisect.json")


def islands(model_id, rev, names):
    """bf16 model with the named submodules of every layer kept in fp32."""
    adapter = load(model_id, rev, "eager", "bfloat16", "cuda")
    for block in adapter.blocks():
        for name in names:
            mod = block.get_submodule(name)
            mod.float()
            mod.register_forward_pre_hook(lambda _m, a, k: (_cast(a, torch.float32), _cast(k, torch.float32)),
                                          with_kwargs=True)
            mod.register_forward_hook(lambda _m, _i, out: _cast(out, torch.bfloat16))
    return adapter


def main(names):
    configure_determinism(0, True)
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names:
        model_id, rev = MODELS[name]
        tok = load_tokenizer(model_id, rev)
        prompts = json.loads(Path(f"evidence/bf16_audit/{name}.json").read_text(encoding="utf-8"))["prompts"]
        cases = []
        for p in prompts:
            ids = torch.tensor([tok.encode(p["corrupt"])], device="cuda")
            a = torch.tensor([tok.encode(" " + p["answer"])[0]], device="cuda")
            b = torch.tensor([tok.encode(" " + p["distractor"])[0]], device="cuda")
            cases.append((ids, a, b))
        ref_model = load(model_id, rev, "eager", "float32", "cuda")
        ref = [float(margins(run(ref_model, ids)[0], a, b)[0]) for ids, a, b in cases]
        del ref_model
        torch.cuda.empty_cache()
        rows = {}
        for label, parts in PARTS.items():
            adapter = islands(model_id, rev, parts)
            err = [abs(float(margins(run(adapter, ids, fp32_logits=True)[0], a, b)[0]) - r)
                   for (ids, a, b), r in zip(cases, ref)]
            rows[label] = {"median_abs_error": float(np.median(err)), "max_abs_error": float(max(err))}
            print(f"{name:12s} fp32 kept for {label:18s} logit diff error median {np.median(err):.3f} "
                  f"max {max(err):.3f}", flush=True)
            del adapter
            torch.cuda.empty_cache()
        result[name] = {"model": model_id, "revision": rev, "n_prompts": len(cases), "parts": rows}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
