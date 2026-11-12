"""outside pythia: does a big invariant key bias hurt bf16/fp16 inference, and does dropping it fix that?

loads a model through transformers directly (no adapter), zeroes the part of each key bias softmax cancels
(all of it with learned or alibi positions, the non-rotary dims with partial rotary), checks in fp32 that
the logits don't move, then measures perplexity in fp32, bf16 and fp16 with and without it.
text: wikitext-2 test, or this repo's own python source for code models.

usage: python scripts/key_bias_other_models.py [name ...]
writes evidence/key_bias_families/inference.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

from kerneltrace.core.manifest import configure_determinism

OUT = Path("evidence/key_bias_families/inference.json")
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            "b08601e04326c79dfdd32d625aee71d232d685c3")
MODELS = {   # name: (repo, revision, text)
    "santacoder": ("bigcode/gpt_bigcode-santacoder", None, "code"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "wikitext"),
    "opt-125m": ("facebook/opt-125m", None, "wikitext"),
    "bloom-560m": ("bigscience/bloom-560m", None, "wikitext"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", None, "wikitext"),
}
SETTINGS = (("fp32", torch.float32, False), ("bf16", torch.bfloat16, False), ("bf16, bias dropped", torch.bfloat16, True),
            ("fp16", torch.float16, False), ("fp16, bias dropped", torch.float16, True))
N_WINDOWS, WINDOW = 32, 512


def text(kind: str) -> str:
    if kind == "code":
        root = Path(__file__).resolve().parents[1]
        return "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((root / "kerneltrace").rglob("*.py")))
    from huggingface_hub import hf_hub_download

    repo, name, rev = WIKITEXT
    return "".join(pd.read_parquet(hf_hub_download(repo, name, repo_type="dataset", revision=rev))["text"].tolist())


@torch.no_grad()
def drop_invariant(model) -> float:
    """zero the softmax-invariant part of every key bias. returns the largest per-head norm removed."""
    cfg, arch, removed = model.config, model.config.model_type, 0.0
    if arch == "gpt_bigcode":
        d, hd = cfg.n_embd, cfg.n_embd // cfg.n_head
        for blk in model.transformer.h:
            kb = blk.attn.c_attn.bias[d:d + hd] if cfg.multi_query else blk.attn.c_attn.bias.view(cfg.n_head, 3, hd)[:, 1]
            removed = max(removed, float(kb.reshape(-1, hd).norm(dim=-1).max())); kb.zero_()
    elif arch == "gpt2":
        d, hd = cfg.n_embd, cfg.n_embd // cfg.n_head
        for blk in model.transformer.h:
            kb = blk.attn.c_attn.bias[d:2 * d]
            removed = max(removed, float(kb.view(-1, hd).norm(dim=-1).max())); kb.zero_()
    elif arch == "opt":
        hd = cfg.hidden_size // cfg.num_attention_heads
        for layer in model.model.decoder.layers:
            kb = layer.self_attn.k_proj.bias
            removed = max(removed, float(kb.view(-1, hd).norm(dim=-1).max())); kb.zero_()
    elif arch == "bloom":
        hd = cfg.hidden_size // cfg.n_head
        for blk in model.transformer.h:
            kb = blk.self_attention.query_key_value.bias.view(cfg.n_head, 3, hd)[:, 1]
            removed = max(removed, float(kb.norm(dim=-1).max())); kb.zero_()
    elif arch == "qwen2":
        # full rotary, so nothing is exactly invariant. the dims that turn under 0.1 rad across 4096
        # positions are close: their part of q.b barely changes from key to key. this is approximate.
        hd = cfg.hidden_size // cfg.num_attention_heads
        inv_freq = cfg.rope_theta ** (-torch.arange(0, hd, 2).float() / hd)
        still = (inv_freq * 4096 < 0.1).nonzero().flatten()
        idx = torch.cat([still, still + hd // 2]).to(model.device)
        for layer in model.model.layers:
            kb = layer.self_attn.k_proj.bias.view(-1, hd)
            removed = max(removed, float(kb[:, idx].norm(dim=-1).max()))
            kb[:, idx] = 0
    else:
        raise ValueError(f"no rule for {arch}")
    return removed


def perplexity(model, ids: torch.Tensor):
    nll, n, nonfinite = 0.0, 0, 0
    for w in range(N_WINDOWS):
        x = ids[w * WINDOW:(w + 1) * WINDOW].unsqueeze(0).cuda()
        if x.shape[1] < WINDOW:
            break
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1).gather(1, x[0, 1:, None])[:, 0]
        if not torch.isfinite(lp).all():
            nonfinite += 1
            continue
        nll -= float(lp.sum()); n += lp.numel()
    return (math.exp(nll / n) if n else float("nan")), nonfinite


def load(repo, rev, dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=dtype, attn_implementation="eager").cuda().eval()


@torch.no_grad()
def run(name):
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    repo, rev, kind = MODELS[name]
    rev = rev or model_info(repo).sha
    configure_determinism(0, True)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    ids = torch.tensor(tok(text(kind))["input_ids"])
    # exactness of the edit in fp32
    m = load(repo, rev, torch.float32)
    x = ids[:WINDOW].unsqueeze(0).cuda()
    before = m(input_ids=x).logits.float()
    removed = drop_invariant(m)
    parity = float((m(input_ids=x).logits.float() - before).abs().max())
    del m; torch.cuda.empty_cache()
    rows = {}
    for label, dtype, drop in SETTINGS:
        m = load(repo, rev, dtype)
        if drop:
            drop_invariant(m)
        ppl, bad = perplexity(m, ids)
        rows[label] = {"perplexity": ppl, "nonfinite_windows": bad}
        del m; torch.cuda.empty_cache()
    ref = rows["fp32"]["perplexity"]
    for r in rows.values():
        r["vs_fp32"] = None if not math.isfinite(r["perplexity"]) else r["perplexity"] / ref - 1
    def cell(v):
        s = "nan" if v["vs_fp32"] is None else f"{v['vs_fp32']:+.1%}"
        return s + (f" ({v['nonfinite_windows']} windows nan)" if v["nonfinite_windows"] else "")

    print(f"{name:12s} removed {removed:7.1f}  fp32 logit change {parity:.4f}  fp32 ppl {ref:.2f}  "
          + "  ".join(f"{k}: {cell(v)}" for k, v in rows.items() if k != "fp32"), flush=True)
    return {"repo": repo, "revision": rev, "text": kind, "windows": N_WINDOWS, "window": WINDOW,
            "invariant_removed_max": removed, "fp32_max_logit_change": parity, "fp32_perplexity": ref, "settings": rows}


def main(names):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names:
        result[name] = run(name)
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
