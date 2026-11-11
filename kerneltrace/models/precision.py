"""keep the precision-sensitive parts of a reduced-precision model in fp32.

some models put a huge offset on every attention score of a query, the same
for every key. softmax subtracts it back out, so in exact arithmetic it does
nothing. pythia-160m reaches offsets past 100,000 with only about 100 of spread
between keys, and bf16 steps by 512 out there, so in bf16 the attention pattern
is decided by rounding. running attention in fp32 and the rest in bf16 keeps
almost all of the memory saving and removes that.

the offset comes from the key bias, in the dimensions rotary embeddings leave
alone. softmax can't see it there, so it gets no gradient and is free to drift
during training. drop_invariant_key_bias zeroes it, which changes nothing in
exact arithmetic and lets the model run in plain bf16.
"""
from __future__ import annotations

import torch

from .adapter import ModelAdapter


def _cast(x, dtype):
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, tuple):
        return tuple(_cast(y, dtype) for y in x)
    if isinstance(x, list):
        return [_cast(y, dtype) for y in x]
    if isinstance(x, dict):
        return {k: _cast(v, dtype) for k, v in x.items()}
    return x


def upcast_attention(adapter: ModelAdapter) -> ModelAdapter:
    """run every attention block in fp32, casting in and back out. returns the adapter."""
    outer = adapter.dtype
    if outer == torch.float32:
        return adapter
    for layer in range(adapter.n_layers):
        mod = adapter.attention(layer)
        mod.float()
        mod.register_forward_pre_hook(
            lambda _m, args, kwargs: (_cast(args, torch.float32), _cast(kwargs, torch.float32)),
            with_kwargs=True)
        mod.register_forward_hook(lambda _m, _inp, out: _cast(out, outer))
    adapter.fp32_attention = True
    return adapter


@torch.no_grad()
def drop_invariant_key_bias(adapter: ModelAdapter) -> float:
    """zero the softmax-invariant part of every key bias. returns the largest norm removed."""
    largest = 0.0
    for layer in range(adapter.n_layers):
        b = adapter.invariant_key_bias(layer)
        rows = b.reshape(-1, b.shape[-1]) if b.dim() > 1 else b.reshape(1, -1)
        largest = max(largest, float(rows.float().norm(dim=-1).max()))
        b.zero_()
    return largest
