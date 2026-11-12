"""bloom: is its bf16 damage another shared score offset, carried by alibi instead of the key bias?

transformers builds bloom's alibi as slope * key_position and adds it to every query's scores. that is
slope * query_position, the same for every key of a query (softmax ignores it), plus slope * (key -
query). the first part grows to about 360 by position 511, where bf16 steps by 2. this writes the
alibi as slope * (key - query) instead, which is the same thing in exact arithmetic, and measures
wikitext-2 perplexity in fp32, bf16 and fp16 both ways.

usage: python scripts/bloom_alibi.py [name ...]   (bloom-560m by default)
writes evidence/key_bias_families/bloom_alibi.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import transformers.models.bloom.modeling_bloom as bloom

from kerneltrace.core.manifest import configure_determinism

sys.path.insert(0, str(Path(__file__).parent))
from key_bias_other_models import WINDOW, drop_invariant, load, perplexity, text  # noqa: E402

OUT = Path("evidence/key_bias_families/bloom_alibi.json")
MODELS = {"bloom-560m": "bigscience/bloom-560m", "bloom-1b1": "bigscience/bloom-1b1"}
SETTINGS = (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16))


def relative(attention_mask, num_heads, dtype):
    a = bloom.build_alibi_tensor(attention_mask, num_heads, torch.float32)  # [b*h, 1, T], slope * key
    return (a - a.transpose(1, 2)).to(dtype)                                # [b*h, T, T], slope * (key - query)


def run(name):
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    repo = MODELS[name]
    rev = model_info(repo).sha
    configure_determinism(0, True)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    ids = torch.tensor(tok(text("wikitext"))["input_ids"])
    original = bloom.BloomModel.build_alibi_tensor
    out = {"repo": repo, "revision": rev, "settings": {}}
    x = ids[:WINDOW].unsqueeze(0).cuda()
    for form in ("as released", "relative alibi", "relative alibi + bias dropped"):
        for label, dtype in SETTINGS:
            if form != "as released":
                bloom.BloomModel.build_alibi_tensor = lambda self, m, h, dtype: relative(m, h, dtype)
            try:
                m = load(repo, rev, dtype)
                if "dropped" in form:
                    drop_invariant(m)
                with torch.no_grad():
                    if dtype == torch.float32:
                        logits = m(input_ids=x, use_cache=False).logits.float()
                        if form == "as released":
                            ref_logits = logits
                        else:
                            out[f"{form}: fp32 max logit change"] = float((logits - ref_logits).abs().max())
                    ppl, bad = perplexity(m, ids)
            finally:
                bloom.BloomModel.build_alibi_tensor = original
            out["settings"][f"{label}, {form}"] = {"perplexity": ppl, "nonfinite_windows": bad}
            del m
            torch.cuda.empty_cache()
    ref = out["settings"]["fp32, as released"]["perplexity"]
    for k, v in out["settings"].items():
        v["vs_fp32"] = v["perplexity"] / ref - 1
        print(f"{name:11s} {k:40s} ppl {v['perplexity']:8.3f}  {v['vs_fp32']:+.2%}  nan windows {v['nonfinite_windows']}",
              flush=True)
    for k in [k for k in out if k.endswith("logit change")]:
        print(f"{name:11s} {k}: {out[k]:.2e}")
    return out


def main(names):
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names:
        result[name] = run(name)
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["bloom-560m"])
