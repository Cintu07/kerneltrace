"""outside pythia: how big are the shared score offsets, and what do bf16 and fp16 do to them?

runs each model in fp32 and catches every attention score matrix where it goes into softmax. for each
query row over the keys it can see:
  offset  = mean score (softmax removes it)
  spread  = max minus min (what softmax uses)
  bf16 damage = total variation distance between softmax(scores) and softmax(scores rounded to bf16)
also the largest q.k before the 1/sqrt(d) scale, which is what overflows fp16 (max 65,504) in models
that scale after the matmul. opt scales the query first and bloom fuses the scale, so there the
matmul output is the scaled score.

each model twice: as released, and with the softmax-invariant key bias zeroed (exact in fp32).
text: 4 windows of 512 tokens, wikitext-2 test or this repo's python source for code models.

usage: python scripts/score_offsets.py [name ...]
writes evidence/key_bias_families/score_offsets.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kerneltrace.core.manifest import configure_determinism

sys.path.insert(0, str(Path(__file__).parent))
from key_bias_other_models import drop_invariant, load, text  # noqa: E402

OUT = Path("evidence/key_bias_families/score_offsets.json")
MODELS = {   # name: (repo, revision, text)
    "pythia-160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46", "wikitext"),
    "santacoder": ("bigcode/gpt_bigcode-santacoder", None, "code"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "wikitext"),
    "opt-125m": ("facebook/opt-125m", None, "wikitext"),
    "bloom-560m": ("bigscience/bloom-560m", None, "wikitext"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", None, "wikitext"),
}
SCALE_AFTER_MATMUL = {"gpt2", "gpt_bigcode", "gpt_neox", "qwen2"}
N_WINDOWS, WINDOW = 4, 512
FP16_MAX = 65504.0


def step(x, mant):
    return 2.0 ** (math.floor(math.log2(abs(x))) - mant) if x else 0.0


class Catcher:
    """wraps F.softmax and summarises every 4-d score tensor that goes through it."""

    def __init__(self, raw_factor: float):
        self.raw_factor = raw_factor
        self.layer = 0
        self.layers: dict[int, dict] = {}
        self.tv = []

    @torch.no_grad()
    def see(self, s):
        s = s.float()[0]  # [heads, T, T]
        T = s.shape[-1]
        vis = torch.ones(T, T, dtype=torch.bool, device=s.device).tril()
        n = vis.sum(-1).float()
        sm = s.masked_fill(~vis, 0.0)
        off = sm.sum(-1) / n
        hi = s.masked_fill(~vis, -float("inf")).amax(-1)
        lo = s.masked_fill(~vis, float("inf")).amin(-1)
        spread = hi - lo
        masked = s.masked_fill(~vis, -float("inf"))
        p = torch.softmax(masked, -1)
        pb = torch.softmax(masked.to(torch.bfloat16).float(), -1)
        tv = 0.5 * (p - pb).abs().sum(-1)                      # [heads, T]
        self.tv.append(tv[:, 1:].flatten().cpu().numpy())       # row 0 sees one key, nothing to round
        h, q = divmod(int(off.abs().flatten().argmax()), T)
        biggest = float(sm.abs().amax())
        rec = self.layers.setdefault(self.layer, {"offset": 0.0, "spread": 0.0, "max_abs_score": 0.0,
                                                  "max_matmul": 0.0, "tv_max": 0.0})
        if abs(float(off[h, q])) > rec["offset"]:
            rec["offset"], rec["spread"] = abs(float(off[h, q])), float(spread[h, q])
        rec["max_abs_score"] = max(rec["max_abs_score"], biggest)
        rec["max_matmul"] = max(rec["max_matmul"], biggest * self.raw_factor)
        rec["tv_max"] = max(rec["tv_max"], float(tv.max()))
        self.layer += 1

    def summary(self):
        tv = np.concatenate(self.tv)
        layers = [{"layer": k, **v, "bf16_step": step(v["offset"], 7)} for k, v in sorted(self.layers.items())]
        peak = max(layers, key=lambda r: r["offset"])
        return {"layers": layers, "largest_offset": peak["offset"], "largest_offset_layer": peak["layer"],
                "spread_there": peak["spread"], "bf16_step_there": peak["bf16_step"],
                "max_matmul": max(r["max_matmul"] for r in layers),
                "layers_over_fp16_max": [r["layer"] for r in layers if r["max_matmul"] > FP16_MAX],
                "rows": int(tv.size), "tv_bf16_median": float(np.median(tv)), "tv_bf16_p99": float(np.quantile(tv, 0.99)),
                "tv_bf16_max": float(tv.max()), "share_tv_bf16_over_0.1": float(np.mean(tv > 0.1)),
                "share_tv_bf16_over_0.01": float(np.mean(tv > 0.01))}


@torch.no_grad()
def measure(model, ids, raw_factor):
    cat = Catcher(raw_factor)
    real = F.softmax

    def spy(x, *a, **k):
        if x.dim() == 4:
            cat.see(x)
        return real(x, *a, **k)

    F.softmax = spy
    try:
        for w in range(N_WINDOWS):
            cat.layer = 0
            model(input_ids=ids[w * WINDOW:(w + 1) * WINDOW].unsqueeze(0).cuda())
    finally:
        F.softmax = real
    return cat.summary()


def run(name):
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    repo, rev, kind = MODELS[name]
    rev = rev or model_info(repo).sha
    configure_determinism(0, True)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    ids = torch.tensor(tok(text(kind))["input_ids"])
    m = load(repo, rev, torch.float32)
    cfg = m.config
    heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    hd = (getattr(cfg, "hidden_size", None) or cfg.n_embd) // heads
    raw_factor = math.sqrt(hd) if cfg.model_type in SCALE_AFTER_MATMUL else 1.0
    out = {"repo": repo, "revision": rev, "text": kind, "windows": N_WINDOWS, "window": WINDOW,
           "head_dim": hd, "matmul_is_unscaled": cfg.model_type in SCALE_AFTER_MATMUL}
    out["as released"] = measure(m, ids, raw_factor)
    out["invariant_removed_max"] = drop_invariant(m)
    out["bias dropped"] = measure(m, ids, raw_factor)
    del m
    torch.cuda.empty_cache()
    for k in ("as released", "bias dropped"):
        r = out[k]
        print(f"{name:13s} {k:12s} offset {r['largest_offset']:9.1f} (L{r['largest_offset_layer']}, spread "
              f"{r['spread_there']:.1f}, bf16 step {r['bf16_step_there']:.3g})  max q.k {r['max_matmul']:9.0f} "
              f"over fp16 in {len(r['layers_over_fp16_max'])} layers  bf16 tv: median {r['tv_bf16_median']:.2e} "
              f"p99 {r['tv_bf16_p99']:.3f} max {r['tv_bf16_max']:.3f} share>0.1 {r['share_tv_bf16_over_0.1']:.2%}",
              flush=True)
    return out


def main(names):
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names:
        result[name] = run(name)
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
