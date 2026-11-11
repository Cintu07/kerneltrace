"""how big is the shared offset on each attention score, next to the part softmax actually uses?

for every query, the scores over the keys it can see are split into their mean (the
offset, which softmax removes) and their max minus min (the spread, which decides the
pattern). the worst head per layer is compared with the bf16 step at the offset. when
the step is bigger than the spread, the pattern in bf16 is set by rounding.

gpt-neox models only, since it hooks their eager attention function.
usage: python scripts/attention_offsets.py
writes evidence/bf16_audit/attention_offsets.json, plus one worked example row in
evidence/bf16_audit/attention_example.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import transformers.models.gpt_neox.modeling_gpt_neox as neox

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.models.adapter import load, load_tokenizer

MODELS = {
    "pythia-160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"),
    "pythia-410m": ("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
    "pythia-1b": ("EleutherAI/pythia-1b", "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"),
}
OUT = Path("evidence/bf16_audit/attention_offsets.json")
EXAMPLE = Path("evidence/bf16_audit/attention_example.json")


def bf16_step(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0.0


def offsets(name):
    model_id, rev = MODELS[name]
    tok = load_tokenizer(model_id, rev)
    prompts = json.loads(Path(f"evidence/bf16_audit/{name}.json").read_text(encoding="utf-8"))["prompts"]
    model = load(model_id, rev, "eager", "float32", "cuda").model
    worst: dict[int, tuple[float, float]] = {}
    original = neox.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling, **kw):
        s = (torch.matmul(query, key.transpose(2, 3)) * scaling)[0]  # [heads, T, T]
        for q in range(2, s.shape[-1]):
            row = s[:, q, : q + 1]
            off = row.mean(-1).abs()
            spread = row.max(-1).values - row.min(-1).values
            h = int(off.argmax())
            if float(off[h]) > worst.get(module.layer_idx, (0.0, 0.0))[0]:
                worst[module.layer_idx] = (float(off[h]), float(spread[h]))
        return original(module, query, key, value, attention_mask, scaling, **kw)

    neox.eager_attention_forward = spy
    try:
        with torch.no_grad():
            for p in prompts:
                model(input_ids=torch.tensor([tok.encode(p["corrupt"])], device="cuda"))
    finally:
        neox.eager_attention_forward = original
    layers = [{"layer": layer, "offset": off, "spread": spread, "bf16_step": bf16_step(off)}
              for layer, (off, spread) in sorted(worst.items())]
    del model
    torch.cuda.empty_cache()
    return {"model": model_id, "revision": rev, "n_prompts": len(prompts), "layers": layers}


def example(name="pythia-160m", layer=10, prompt=0):
    """one query's scores in the worst head of one layer, before and after bf16 rounding."""
    model_id, rev = MODELS[name]
    tok = load_tokenizer(model_id, rev)
    text = json.loads(Path(f"evidence/bf16_audit/{name}.json").read_text(encoding="utf-8"))["prompts"][prompt]["corrupt"]
    ids = tok.encode(text)
    model = load(model_id, rev, "eager", "float32", "cuda").model
    got = {}
    original = neox.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling, **kw):
        if module.layer_idx == layer:
            row = (torch.matmul(query, key.transpose(2, 3)) * scaling)[0, :, -1, :]
            h = int(row.mean(-1).abs().argmax())
            rounded = row[h].to(torch.bfloat16).float()
            got.update(head=h, scores=row[h].tolist(), scores_bf16=rounded.tolist(),
                       attention=torch.softmax(row[h], -1).tolist(),
                       attention_bf16=torch.softmax(rounded, -1).tolist())
        return original(module, query, key, value, attention_mask, scaling, **kw)

    neox.eager_attention_forward = spy
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"))
    finally:
        neox.eager_attention_forward = original
    return {"model": model_id, "revision": rev, "layer": layer, "prompt": text,
            "tokens": [tok.decode([t]) for t in ids], **got}


def main():
    configure_determinism(0, True)
    EXAMPLE.write_text(json.dumps(example(), indent=1), encoding="utf-8", newline="\n")
    result ={name: offsets(name) for name in MODELS}
    for name, r in result.items():
        bad = [x for x in r["layers"] if x["bf16_step"] > x["spread"]]
        peak = max(r["layers"], key=lambda x: x["offset"])
        print(f"{name:12s} largest offset {peak['offset']:9.0f} (layer {peak['layer']}, spread {peak['spread']:.0f}, "
              f"bf16 step {peak['bf16_step']:.0f})  layers where the step beats the spread: {len(bad)}/{len(r['layers'])}")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
