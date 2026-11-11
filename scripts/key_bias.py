"""where pythia's attention offsets come from, and what removing them costs.

parity: in fp32, zero the softmax-invariant part of every key bias and compare the
logits with the untouched model, over the whole vocabulary at every position, on the
audit's prompts. in exact arithmetic they are identical.

training: read the key bias straight out of pythia-160m's training checkpoints and
track how big the invariant part (the dimensions rotary leaves alone) and the rotary
part get. the invariant part gets no gradient in exact arithmetic.

suite: the same numbers for the final checkpoint of every pythia size, 70m to 12b.

prescale: why pythia-1.4b still overflows fp16 with the bias dropped.

usage: python scripts/key_bias.py [parity] [training] [suite] [prescale]
writes evidence/bf16_audit/key_bias.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.interventions.patching import run
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.models.precision import drop_invariant_key_bias

MODELS = {
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "pythia-160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46"),
    "pythia-410m": ("EleutherAI/pythia-410m", "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
    "pythia-1b": ("EleutherAI/pythia-1b", "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"),
}
STEPS = (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000)
OUT = Path("evidence/bf16_audit/key_bias.json")


@torch.no_grad()
def parity(name):
    model_id, rev = MODELS[name]
    tok = load_tokenizer(model_id, rev)
    prompts = json.loads(Path(f"evidence/bf16_audit/{name}.json").read_text(encoding="utf-8"))["prompts"]
    a = load(model_id, rev, "eager", "float32", "cuda")
    b = load(model_id, rev, "eager", "float32", "cuda")
    removed = drop_invariant_key_bias(b)
    worst, worst_rel = 0.0, 0.0
    for p in prompts:
        ids = torch.tensor([tok.encode(p["corrupt"])], device="cuda")
        la, lb = run(a, ids)[0], run(b, ids)[0]
        worst = max(worst, float((la - lb).abs().max()))
        worst_rel = max(worst_rel, float(((la - lb).abs() / la.abs().clamp_min(1.0)).max()))
    del a, b
    torch.cuda.empty_cache()
    print(f"{name:12s} largest invariant key bias norm removed {removed:7.1f}   "
          f"largest fp32 logit change {worst:.2e} (relative {worst_rel:.1e}) over {len(prompts)} prompts", flush=True)
    return {"largest_norm_removed": removed, "max_abs_logit_change": worst, "max_rel_logit_change": worst_rel,
            "n_prompts": len(prompts)}


def _remote_tensors(repo, revision, names):
    """read a few tensors out of a remote model.safetensors with range requests, not the whole file."""
    import struct

    import numpy as np
    import requests
    from huggingface_hub import hf_hub_url

    def fetch(filename, a, b):
        url = hf_hub_url(repo, filename, revision=revision)
        r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        return r.content

    try:  # sharded checkpoints list which file holds each tensor
        index = requests.get(hf_hub_url(repo, "model.safetensors.index.json", revision=revision), timeout=60)
        where = index.json()["weight_map"] if index.ok else {}
    except ValueError:
        where = {}
    headers = {}

    def header(filename):
        if filename not in headers:
            n = struct.unpack("<Q", fetch(filename, 0, 7))[0]
            headers[filename] = (n, json.loads(fetch(filename, 8, 8 + n - 1)))
        return headers[filename]

    kinds = {"F32": np.float32, "F16": np.float16}
    out = {}
    for name in names:
        filename = where.get(name, "model.safetensors")
        n, head = header(filename)
        meta = head[name]
        get = lambda a, b, f=filename: fetch(f, a, b)
        a, b = meta["data_offsets"]
        raw = get(8 + n + a, 8 + n + b - 1)
        if meta["dtype"] == "BF16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        else:
            t = torch.from_numpy(np.frombuffer(raw, dtype=kinds[meta["dtype"]]).copy())
        out[name] = t.float().view(meta["shape"])
    return out


def checkpoint_bias(step, repo="EleutherAI/pythia-160m"):
    """per layer, the largest per-head norm of the invariant and the rotary part of the key bias."""
    from huggingface_hub import HfApi
    from transformers import GPTNeoXConfig

    sha = HfApi().model_info(repo, revision=f"step{step}" if step is not None else None).sha
    cfg = GPTNeoXConfig.from_pretrained(repo, revision=sha)
    heads = cfg.num_attention_heads
    size = cfg.hidden_size // heads
    rot = int(size * cfg.rotary_pct)
    names = [f"gpt_neox.layers.{layer}.attention.query_key_value.bias" for layer in range(cfg.num_hidden_layers)]
    state = _remote_tensors(repo, sha, names)
    layers = []
    for name in names:
        bias = state[name].view(heads, 3, size)
        key, query = bias[:, 1], bias[:, 0]
        layers.append({"invariant": float(key[:, rot:].norm(dim=-1).max()),
                       "rotary": float(key[:, :rot].norm(dim=-1).max()),
                       "query": float(query.norm(dim=-1).max())})
    return sha, layers


def training():
    out = {}
    for step in STEPS:
        sha, layers = checkpoint_bias(step)
        out[step] = {"revision": sha, "layers": layers}
        print(f"step {step:6d}  largest key bias norm, invariant part {max(l['invariant'] for l in layers):7.1f}   "
              f"rotary part {max(l['rotary'] for l in layers):6.1f}   query bias {max(l['query'] for l in layers):6.1f}",
              flush=True)
    return out


SUITE = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")


def suite():
    """the final checkpoint of every pythia size, read without downloading the weights."""
    out = {}
    for size in SUITE:
        sha, layers = checkpoint_bias(None, f"EleutherAI/pythia-{size}")
        inv = max(l["invariant"] for l in layers)
        out[size] = {"revision": sha, "invariant_max": inv, "rotary_max": max(l["rotary"] for l in layers),
                     "query_max": max(l["query"] for l in layers), "layers": layers}
        print(f"pythia-{size:5s} largest key bias norm, invariant part {inv:7.1f}   "
              f"rotary part {out[size]['rotary_max']:6.1f}   query bias {out[size]['query_max']:6.1f}", flush=True)
    return out


def prescale(model_id="EleutherAI/pythia-1.4b", rev="fedc38a16eea3bd36a96b906d78d11d2ce18ed79"):
    """with the bias dropped, how big does q . k get before the attention code scales it?

    the eager attention multiplies q by k first and scales after, so in fp16 that raw product
    is what has to fit under 65,504. measured in fp32 on the first wikitext window.
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as neox

    sys.path.insert(0, str(Path(__file__).parent))
    from lm_quality import WINDOW, wikitext

    tok = load_tokenizer(model_id, rev)
    ids = torch.tensor([tok.encode(wikitext()[:40_000])[:WINDOW]], device="cuda")
    adapter = load(model_id, rev, "eager", "float32", "cuda")
    drop_invariant_key_bias(adapter)
    seen = {"raw": 0.0, "scaled": 0.0}
    original = neox.eager_attention_forward

    def spy(module, query, key, value, attention_mask, scaling, **kw):
        raw = torch.matmul(query, key.transpose(2, 3))
        seen["raw"] = max(seen["raw"], float(raw.abs().max()))
        seen["scaled"] = max(seen["scaled"], float((raw * scaling).abs().max()))
        return original(module, query, key, value, attention_mask, scaling, **kw)

    neox.eager_attention_forward = spy
    try:
        with torch.no_grad():
            adapter.model(input_ids=ids)
    finally:
        neox.eager_attention_forward = original
    print(f"{model_id} with the bias dropped: largest q.k before scaling {seen['raw']:,.0f}, "
          f"after {seen['scaled']:,.0f} (fp16 tops out at 65,504)", flush=True)
    return {"model": model_id, "revision": rev, "max_raw_qk": seen["raw"], "max_scaled_score": seen["scaled"]}


def main(parts):
    configure_determinism(0, True)
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if "parity" in parts:
        result["parity"] = {name: parity(name) for name in MODELS}
    if "suite" in parts:
        result["suite"] = suite()
    if "prescale" in parts:
        result["prescale"] = prescale()
    if "training" in parts:
        result["training"] = {"model": "EleutherAI/pythia-160m", "steps": training()}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["parity", "training", "suite", "prescale"])
