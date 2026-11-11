"""one prompt, two batch sizes. in fp32 the logit diff doesn't move. in bf16 it jumps by 0.5.

gpt-2's logits sit around 80 to 140 in magnitude, where bf16 can only step by 0.5. the
same prompt alone and as row 0 of a batch of 15 copies takes slightly different kernels,
and that is enough to land on a different step. the full sweep is scripts/bf16_audit.py.

usage: python scripts/bf16_batch_check.py
"""
import json

import torch

from kerneltrace.core.manifest import configure_determinism
from kerneltrace.interventions.patching import margins, run
from kerneltrace.models.adapter import load, load_tokenizer

MODEL = "openai-community/gpt2"
REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

configure_determinism(0, True)
tok = load_tokenizer(MODEL, REV)
prompt = json.load(open("evidence/causal_trace/gpt2_ioi_trace.json", encoding="utf-8"))["prompts"][41]
ids = torch.tensor([tok.encode(prompt["corrupt"])], device="cuda")
a = torch.tensor([tok.encode(" " + prompt["answer"])[0]], device="cuda")
b = torch.tensor([tok.encode(" " + prompt["distractor"])[0]], device="cuda")
T = ids.shape[1]

for dtype, fp32_logits in (("float32", False), ("bfloat16", False), ("bfloat16", True)):
    adapter = load(MODEL, REV, "eager", dtype, "cuda")
    alone = margins(run(adapter, ids, fp32_logits=fp32_logits)[0], a, b)[0].item()
    batched = margins(run(adapter, ids.repeat(T, 1), fp32_logits=fp32_logits)[0], a.repeat(T), b.repeat(T))[0].item()
    label = dtype + (" + fp32 head" if fp32_logits else "")
    print(f"{label:21s} alone {alone:+.4f}   in a batch of {T} {batched:+.4f}   gap {abs(alone - batched):.4f}")
