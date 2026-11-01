"""Evidence bundle: one directory per run, every file hashed at the end.

Text is always written with LF line endings so a bundle hashes the same on every OS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.manifest import sha256_file


def _clean(obj):
    """Make results JSON-safe: numpy scalars, tuples as keys, NaN as null."""
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class Bundle:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def json(self, rel: str, obj) -> Path:
        p = self.path(rel)
        p.write_text(json.dumps(_clean(obj), indent=2, default=str), encoding="utf-8", newline="\n")
        return p

    def jsonl(self, rel: str, rows) -> Path:
        p = self.path(rel)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(_clean(r), default=str) + "\n")
        return p

    def text(self, rel: str, text: str) -> Path:
        p = self.path(rel)
        p.write_text(text, encoding="utf-8", newline="\n")
        return p

    def parquet(self, rel: str, df: pd.DataFrame) -> Path:
        p = self.path(rel)
        df.to_parquet(p, index=False)
        df.to_csv(p.with_suffix(".csv"), index=False, lineterminator="\n")
        return p

    def file_hashes(self) -> dict:
        return {f.relative_to(self.root).as_posix(): sha256_file(f)
                for f in sorted(self.root.rglob("*")) if f.is_file() and f.name != "files.json"}
