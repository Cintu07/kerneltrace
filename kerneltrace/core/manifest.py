"""run provenance. everything you need to tell whether two runs can be compared."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def configure_determinism(seed: int, deterministic: bool) -> dict:
    """seed everything and pin the numerical switches. returns what it set."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_tf32(False)
    if deterministic:
        # cublas needs this set before the first gemm, or it won't be deterministic.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return numerical_state() | {"seed": seed}


def numerical_state() -> dict:
    return {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "sdp_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdp_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdp_math_enabled": torch.backends.cuda.math_sdp_enabled(),
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    """software, hardware and code state, for the manifest."""
    installed = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    status = _git("status", "--porcelain") or ""
    dirty = [line.split(maxsplit=1)[1] for line in status.splitlines() if line.strip()]
    env = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "transformers": _pkg("transformers"),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_driver": _driver_version(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_dirty_paths": dirty,
        "packages_sha256": sha256_json(installed),
        "packages": installed,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env |= {
            "gpu_name": props.name,
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_memory_gb": round(props.total_memory / 2**30, 2),
        }
    return env


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("experiment requests CUDA but torch reports no CUDA device")
    return requested


def new_run_id(experiment_id: str, spec_hash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{experiment_id}_{spec_hash[:8]}"
