"""exploratory: where does bloom-560m lose 13% perplexity in bf16? (no prediction was written for this.)

bf16 everywhere, except one group of modules kept in fp32 (cast in, cast back out). fp16 is fine on
this model (+0.2%), so the loss comes from bf16's 8-bit significand, not from range.

usage: python scripts/bloom_bisect.py
writes evidence/key_bias_families/bloom_bisect.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.models.precision import _cast

sys.path.insert(0, str(Path(__file__).parent))
from key_bias_other_models import load, perplexity, text  # noqa: E402

OUT = Path("evidence/key_bias_families/bloom_bisect.json")
REPO = "bigscience/bloom-560m"


def keep_fp32(modules):
    for mod in modules:
        mod.float()
        mod.register_forward_pre_hook(lambda _m, a, k: (_cast(a, torch.float32), _cast(k, torch.float32)),
                                      with_kwargs=True)
        mod.register_forward_hook(lambda _m, _i, out: _cast(out, torch.bfloat16))


GROUPS = {
    "nothing": lambda m: [],
    "attention": lambda m: [b.self_attention for b in m.transformer.h],
    "mlp": lambda m: [b.mlp for b in m.transformer.h],
    "layernorms": lambda m: ([b.input_layernorm for b in m.transformer.h] + [b.post_attention_layernorm for b in m.transformer.h]
                             + [m.transformer.ln_f, m.transformer.word_embeddings_layernorm]),
    "embedding layernorm": lambda m: [m.transformer.word_embeddings_layernorm],
    "lm head": "head",
}


def fp32_head(m):
    """the head shares its weight with the embedding, so give it its own fp32 copy instead of casting it."""
    w = m.lm_head.weight.detach().float()
    m.lm_head.forward = lambda h: (h.float() @ w.T)



def main():
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    rev = model_info(REPO).sha
    configure_determinism(0, True)
    tok = AutoTokenizer.from_pretrained(REPO, revision=rev)
    ids = torch.tensor(tok(text("wikitext"))["input_ids"])
    with torch.no_grad():
        m = load(REPO, rev, torch.float32)
        ref = perplexity(m, ids)[0]
        del m
        out = {"repo": REPO, "revision": rev, "fp32": ref, "bf16 with fp32": {}}
        for name, pick in GROUPS.items():
            m = load(REPO, rev, torch.bfloat16)
            if pick == "head":
                fp32_head(m)
            else:
                keep_fp32(pick(m))
            ppl = perplexity(m, ids)[0]
            out["bf16 with fp32"][name] = {"perplexity": ppl, "vs_fp32": ppl / ref - 1}
            print(f"bf16, {name:20s} in fp32: ppl {ppl:.3f}  {ppl / ref - 1:+.2%}", flush=True)
            del m
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
