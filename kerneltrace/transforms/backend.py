"""Numerical execution conditions and evidence of what actually ran.

Requesting `attn_implementation="sdpa"` says nothing about which kernel SDPA
dispatches to. Each condition is therefore probed with the profiler and the
observed attention path is recorded. If a condition forces a kernel and the
profiler does not see it, the condition is marked invalid instead of being
compared as if it had run.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch.profiler import ProfilerActivity, profile

from ..core.manifest import set_tf32
from ..core.spec import ConditionSpec

# Kernel families recognised in profiler op names, most specific first.
SDPA_OPS = {
    "flash": ("_scaled_dot_product_flash_attention", "_flash_attention_forward"),
    "efficient": ("_scaled_dot_product_efficient_attention", "_efficient_attention_forward"),
    "cudnn": ("_scaled_dot_product_cudnn_attention",),
    "math": ("_scaled_dot_product_attention_math",),
    "flash_cpu": ("_scaled_dot_product_flash_attention_for_cpu",),
}


def _sdpa_backend(name: str):
    from torch.nn.attention import SDPBackend

    return {
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[name]


@contextmanager
def execution(condition: ConditionSpec):
    """Apply the condition's TF32 and SDPA-kernel settings for the duration."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision())
    set_tf32(condition.tf32)
    if condition.sdpa_kernel is not None:
        from torch.nn.attention import sdpa_kernel

        kernel_ctx = sdpa_kernel(_sdpa_backend(condition.sdpa_kernel))
    else:
        kernel_ctx = nullcontext()
    try:
        with kernel_ctx:
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev[0]
        torch.backends.cudnn.allow_tf32 = prev[1]
        torch.set_float32_matmul_precision(prev[2])


def observed_attention(op_names: set[str]) -> str:
    """Name the attention path seen in a set of profiler op names."""
    found = [fam for fam, keys in SDPA_OPS.items()
             if any(k in name for name in op_names for k in keys)]
    if found:
        return "+".join(sorted(found))
    if any("scaled_dot_product" in n for n in op_names):
        return "sdpa_unknown"
    if any(n in ("aten::softmax", "aten::_softmax") for n in op_names):
        return "eager_matmul_softmax"
    return "unknown"


def probe(adapter, input_ids: torch.Tensor, condition: ConditionSpec) -> dict:
    """Profile one forward and decide whether the condition ran as requested."""
    activities = [ProfilerActivity.CPU]
    if input_ids.is_cuda:
        activities.append(ProfilerActivity.CUDA)
    with execution(condition), torch.no_grad():
        with profile(activities=activities) as prof:
            adapter.model(input_ids=input_ids)
    events = prof.key_averages()
    names = {e.key for e in events}
    observed = observed_attention(names)
    problems = []
    if adapter.attn_implementation != condition.attn_implementation:
        problems.append(f"model reports attn_implementation={adapter.attn_implementation}")
    if condition.attn_implementation == "eager" and any("scaled_dot_product" in n for n in names):
        problems.append("eager condition dispatched an SDPA op")
    if condition.attn_implementation == "sdpa" and observed == "eager_matmul_softmax":
        problems.append("sdpa condition shows no SDPA op")
    if condition.sdpa_kernel is not None and condition.sdpa_kernel not in observed.split("+"):
        problems.append(f"forced {condition.sdpa_kernel} kernel not observed (saw {observed})")
    attention_like = sorted(n for n in names if any(
        k in n.lower() for k in ("attention", "fmha", "flash", "softmax", "sdpa")))
    gemm_like = sorted(n for n in names if any(k in n.lower() for k in ("gemm", "cutlass", "sm80", "sm90", "sm100", "sm120")))
    return {
        "condition": condition.name,
        "requested": condition.model_dump(),
        "observed_attention": observed,
        "verified": not problems,
        "problems": problems,
        "attention_ops": attention_like,
        "gemm_kernels": gemm_like[:50],
        "n_distinct_ops": len(names),
        "tf32_matmul_during_run": condition.tf32,
    }


def factors(reference: ConditionSpec, condition: ConditionSpec) -> list[str]:
    """Which experimental factors differ between two conditions."""
    out = []
    if (reference.attn_implementation, reference.sdpa_kernel) != (
            condition.attn_implementation, condition.sdpa_kernel):
        out.append("attention_backend")
    if reference.dtype != condition.dtype:
        out.append("dtype")
    if reference.tf32 != condition.tf32:
        out.append("tf32")
    if reference.batch_size != condition.batch_size:
        out.append("batching")
    return out or ["repeat"]
