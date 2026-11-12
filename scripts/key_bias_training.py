"""why does the softmax-invariant key bias grow? train a small pythia-shaped model from scratch and watch.

the invariant part of a head's key bias adds q.b to every score of a query, softmax cancels it, and its
exact gradient is (sum over keys of the score gradient) times q, which is zero. this trains the same small
gpt-neox model under fp32, fp16 and bf16 autocast, with adam at two epsilons and with sgd, and logs the
invariant part, the rotary part of the same tensor, their gradients, and the size of each optimizer step.

predictions: experiments/key_bias_training_prediction.txt (committed before the first run).

usage: python scripts/key_bias_training.py [run ...]
writes evidence/key_bias_training/<run>.json and summary.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.models.adapter import load_tokenizer

TOKENIZER = ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46")
WIKITEXT_REPO, WIKITEXT_REV = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"
OUT = Path("evidence/key_bias_training")

MODEL = dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024,
             rotary_pct=0.25, use_parallel_residual=True, max_position_embeddings=512)
TRAIN = dict(steps=3000, batch=16, seq=256, lr=1e-3, warmup=100, min_lr_frac=0.1, betas=(0.9, 0.95),
             weight_decay=0.01, clip=1.0, seed=0, log_every=50, eval_every=500, eval_batches=20)
# name: (autocast dtype or None, optimizer, eps or lr override)
RUNS = {
    "fp32 adam": (None, "adam", 1e-8),
    "fp16 adam": (torch.float16, "adam", 1e-8),
    "bf16 adam": (torch.bfloat16, "adam", 1e-8),
    "fp16 adam eps 1e-4": (torch.float16, "adam", 1e-4),
    "fp16 sgd": (torch.float16, "sgd", 0.1),
    "fp16 adam fp32 attention": (torch.float16, "adam", 1e-8),
}
FP32_ATTENTION = {"fp16 adam fp32 attention"}


def attention_in_fp32(model):
    """run every attention module with autocast off and fp32 inputs; the rest stays in autocast."""
    for layer in model.gpt_neox.layers:
        att = layer.attention
        orig = att.forward

        def forward(*args, _orig=orig, **kwargs):
            cast = lambda t: t.float() if torch.is_tensor(t) and t.is_floating_point() else t
            args = [cast(a) for a in args]
            kwargs = {k: (tuple(cast(x) for x in v) if isinstance(v, tuple) else cast(v)) for k, v in kwargs.items()}
            with torch.autocast("cuda", enabled=False):
                return _orig(*args, **kwargs)

        att.forward = forward


def tokens(split: str, tok) -> torch.Tensor:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKITEXT_REPO, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=WIKITEXT_REV)
    text = "".join(pd.read_parquet(path)["text"].tolist())
    return torch.tensor(tok(text)["input_ids"], dtype=torch.long)


def build(vocab: int):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    cfg = GPTNeoXConfig(vocab_size=vocab, attn_implementation="eager", **MODEL)
    return GPTNeoXForCausalLM(cfg)


def bias_parts(model):
    """per layer: (invariant, rotary, query) views of the qkv bias, each [heads, dims]."""
    out = []
    for layer in model.gpt_neox.layers:
        att = layer.attention
        b = att.query_key_value.bias.view(att.config.num_attention_heads, 3, att.head_size)
        out.append((b[:, 1, att.rotary_ndims:], b[:, 1, : att.rotary_ndims], b[:, 0, :]))
    return out


def grad_parts(model):
    out = []
    for layer in model.gpt_neox.layers:
        att = layer.attention
        g = att.query_key_value.bias.grad.view(att.config.num_attention_heads, 3, att.head_size)
        out.append((g[:, 1, att.rotary_ndims:], g[:, 1, : att.rotary_ndims]))
    return out


def optimizer(model, kind: str, value: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": TRAIN["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if kind == "adam":
        return torch.optim.AdamW(groups, lr=TRAIN["lr"], betas=TRAIN["betas"], eps=value)
    return torch.optim.SGD(groups, lr=value, momentum=0.9)


def lr_at(step: int, base: float) -> float:
    if step < TRAIN["warmup"]:
        return base * (step + 1) / TRAIN["warmup"]
    t = (step - TRAIN["warmup"]) / max(1, TRAIN["steps"] - TRAIN["warmup"])
    return base * (TRAIN["min_lr_frac"] + (1 - TRAIN["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * t)))


def batch_at(data: torch.Tensor, step: int, gen: torch.Generator):
    starts = torch.randint(0, len(data) - TRAIN["seq"] - 1, (TRAIN["batch"],), generator=gen)
    x = torch.stack([data[s: s + TRAIN["seq"]] for s in starts.tolist()])
    return x.cuda(non_blocking=True)


@torch.no_grad()
def evaluate(model, val: torch.Tensor, dtype) -> float:
    model.eval()
    n, seq, losses = TRAIN["eval_batches"], TRAIN["seq"], []
    for i in range(n):
        x = val[i * TRAIN["batch"] * seq: (i + 1) * TRAIN["batch"] * seq].view(TRAIN["batch"], seq).cuda()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            losses.append(float(model(input_ids=x, labels=x).loss))
    model.train()
    return float(np.mean(losses))


def train(name: str, train_ids: torch.Tensor, val_ids: torch.Tensor, vocab: int) -> dict:
    dtype, kind, value = RUNS[name]
    configure_determinism(TRAIN["seed"], False)
    model = build(vocab).cuda().train()
    if name in FP32_ATTENTION:
        attention_in_fp32(model)
    opt = optimizer(model, kind, value)
    base_lr = TRAIN["lr"] if kind == "adam" else value
    scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)
    gen = torch.Generator().manual_seed(TRAIN["seed"])
    log, t0 = [], time.time()
    for step in range(TRAIN["steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, base_lr)
        x = batch_at(train_ids, step, gen)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["clip"])
        record = step % TRAIN["log_every"] == 0 or step == TRAIN["steps"] - 1
        if record:
            grads = [(float(gi.norm()), float(gr.norm())) for gi, gr in grad_parts(model)]
            before = [(i.detach().clone(), r.detach().clone()) for i, r, _ in bias_parts(model)]
        scaler.step(opt)
        scaler.update()
        if not math.isfinite(float(loss)):
            log.append({"step": step, "loss": float(loss), "diverged": True})
            break
        if record:
            parts = bias_parts(model)
            row = {"step": step, "loss": float(loss), "lr": lr_at(step, base_lr),
                   "invariant": [float(i.norm()) for i, _, _ in parts],
                   "rotary": [float(r.norm()) for _, r, _ in parts],
                   "query": [float(q.norm()) for _, _, q in parts],
                   "grad_invariant": [g[0] for g in grads], "grad_rotary": [g[1] for g in grads],
                   "step_invariant": [float((i - bi).norm()) for (i, _, _), (bi, _) in zip(parts, before)],
                   "step_rotary": [float((r - br).norm()) for (_, r, _), (_, br) in zip(parts, before)]}
            if step % TRAIN["eval_every"] == 0 or step == TRAIN["steps"] - 1:
                row["val_loss"] = evaluate(model, val_ids, dtype)
            log.append(row)
            if step % 500 == 0 or step == TRAIN["steps"] - 1:
                print(f"{name:20s} step {step:5d} loss {float(loss):.3f}  invariant {max(row['invariant']):8.3f}"
                      f"  rotary {max(row['rotary']):7.3f}  {time.time() - t0:5.0f}s", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return {"run": name, "autocast": str(dtype), "optimizer": kind, "eps_or_lr": value,
            "model": MODEL, "train": TRAIN, "log": log}


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(*TOKENIZER)
    train_ids, val_ids = tokens("train", tok), tokens("test", tok)
    vocab = 50304   # pythia's padded vocab
    print(f"wikitext-2: {len(train_ids):,} train tokens, {len(val_ids):,} test tokens", flush=True)
    for name in names:
        result = train(name, train_ids, val_ids, vocab)
        path = OUT / (name.replace(" ", "_").replace("-", "m") + ".json")
        path.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(RUNS))
