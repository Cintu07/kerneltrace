"""does the key bias problem reach ordinary use of the model, outside patching?

perplexity on the start of the wikitext-2 test split, in fp32, bf16 and fp16, with and
without the softmax-invariant key bias, and with attention kept in fp32. plus how many of the audit's ioi prompts give
non-finite logits in fp16, where the largest finite value is 65,504.

usage: python scripts/lm_quality.py [n_windows] [model ...]
writes evidence/bf16_audit/lm_quality.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.interventions.patching import fp32_unembed
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.models.precision import drop_invariant_key_bias, upcast_attention

MODELS = {
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "pythia-160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"),
    "pythia-410m": ("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
    "pythia-1b": ("EleutherAI/pythia-1b", "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"),
    "pythia-70m": ("EleutherAI/pythia-70m", "a39f36b100fe8a5377810d56c3f4789b9c53ac42"),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"),
}
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            "b08601e04326c79dfdd32d625aee71d232d685c3")
# label, dtype, invariant key bias dropped, unembedding in fp32, attention in fp32
SETTINGS = (("fp32", "float32", False, False, False), ("bf16", "bfloat16", False, False, False),
            ("bf16, key bias dropped", "bfloat16", True, False, False),
            ("bf16, key bias dropped + fp32 head", "bfloat16", True, True, False),
            ("bf16 + fp32 attention and head", "bfloat16", False, True, True),
            ("fp16", "float16", False, False, False), ("fp16, key bias dropped", "float16", True, False, False),
            ("fp16 + fp32 attention", "float16", False, False, True))
WINDOW = 512
OUT = Path("evidence/bf16_audit/lm_quality.json")


def wikitext():
    from huggingface_hub import hf_hub_download

    repo, name, rev = WIKITEXT
    path = hf_hub_download(repo, name, repo_type="dataset", revision=rev)
    return "".join(pd.read_parquet(path)["text"].tolist())


@torch.no_grad()
def perplexity(adapter, windows, fp32_head=False):
    nll, count, bad = 0.0, 0, 0
    for w in windows:
        logits = (fp32_unembed(adapter, w[None]) if fp32_head else adapter.model(input_ids=w[None]).logits)[0, :-1].float()
        if not torch.isfinite(logits).all():
            bad += 1
            continue
        nll += float(torch.nn.functional.cross_entropy(logits, w[1:], reduction="sum"))
        count += w.numel() - 1
    return (math.exp(nll / count) if count else float("nan")), bad


@torch.no_grad()
def nonfinite_prompts(adapter, tok, prompts):
    bad = 0
    for p in prompts:
        logits = adapter.model(input_ids=torch.tensor([tok.encode(p["corrupt"])], device=adapter.device)).logits
        bad += int(not torch.isfinite(logits).all())
    return bad


def main(n_windows, names):
    configure_determinism(0, True)
    text = wikitext()
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    result.update({"wikitext": dict(zip(("repo", "file", "revision"), WIKITEXT)), "window": WINDOW,
                   "n_windows": n_windows})
    result.setdefault("models", {})
    for name in names:
        model_id, rev = MODELS[name]
        tok = load_tokenizer(model_id, rev)
        ids = torch.tensor(tok.encode(text[:400_000]), device="cuda")
        windows = [ids[i * WINDOW:(i + 1) * WINDOW] for i in range(n_windows)]
        audit = Path(f"evidence/bf16_audit/{name}.json")  # only the audited models have ioi prompts
        prompts = json.loads(audit.read_text(encoding="utf-8"))["prompts"] if audit.exists() else []
        rows = {}
        for label, dtype, drop, fp32_head, fp32_attention in SETTINGS:
            adapter = load(model_id, rev, "eager", dtype, "cuda")
            if drop:
                drop_invariant_key_bias(adapter)
            if fp32_attention:
                upcast_attention(adapter)
            ppl, bad = perplexity(adapter, windows, fp32_head)
            rows[label] = {"perplexity": ppl, "nonfinite_windows": bad,
                           "nonfinite_prompts": nonfinite_prompts(adapter, tok, prompts)}
            del adapter
            torch.cuda.empty_cache()
        ref = rows["fp32"]["perplexity"]
        for label, r in rows.items():
            r["vs_fp32"] = r["perplexity"] / ref - 1 if r["perplexity"] == r["perplexity"] else None
            change = f"{r['vs_fp32']:+.2%}" if r["vs_fp32"] is not None else "   n/a"
            print(f"{name:12s} {label:34s} perplexity {r['perplexity']:8.3f} ({change})  "
                  f"non-finite: {r['nonfinite_windows']}/{n_windows} windows, "
                  f"{r['nonfinite_prompts']}/{len(prompts)} ioi prompts", flush=True)
        result["models"][name] = rows
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64, sys.argv[2:] or list(MODELS))
