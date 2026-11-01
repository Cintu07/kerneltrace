"""effect sizes, uncertainty and stability metrics.

bootstrap resampling is clustered by template. prompts from one template are
correlated evidence, so the template is the thing that gets resampled.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, spearmanr


def relative_restoration(clean, corrupt, patched, epsilon: float = 1e-5):
    """(patched - corrupt) / (clean - corrupt), or None when the contrast is too small."""
    denominator = clean - corrupt
    if abs(denominator) < epsilon:
        return None
    return (patched - corrupt) / denominator


def restoration_array(clean: np.ndarray, corrupt: np.ndarray, patched: np.ndarray) -> np.ndarray:
    """vectorised restoration. callers must have gated cases on min_contrast already."""
    contrast = clean - corrupt
    if np.any(np.abs(contrast) < 1e-5):
        raise ValueError("restoration requested for a case with no contrast")
    return (patched - corrupt) / contrast


def _cluster_index(clusters: np.ndarray):
    labels = np.unique(clusters)
    return labels, [np.flatnonzero(clusters == lab) for lab in labels]


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int,
                      stat=np.mean) -> np.ndarray:
    """bootstrap distribution of `stat` over the first axis, resampling whole clusters.

    `values` can be [N] or [N, K], and the statistic runs along axis 0.
    """
    values = np.asarray(values, dtype=float)
    labels, members = _cluster_index(np.asarray(clusters))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(labels), size=len(labels))
        idx = np.concatenate([members[i] for i in pick])
        out.append(stat(values[idx], axis=0))
    return np.asarray(out)


def ci(boot: np.ndarray, level: float = 0.95) -> tuple:
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return lo, hi


def summarize(values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    boot = cluster_bootstrap(values, clusters, n_boot, seed)
    lo, hi = ci(boot)
    return {
        "n": int(values.size),
        "n_clusters": int(len(np.unique(clusters))),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(1 - (1 - level) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(spearmanr(a, b).statistic)


def topk_overlap(a, b, k: int = 3):
    """jaccard overlap of the top-k indices by absolute effect. kept from v0.1 so old numbers still compare."""
    if len(a) != len(b):
        raise ValueError("Effect vectors must have equal length")
    if not len(a):
        return None
    k = min(k, len(a))
    sa = set(sorted(range(len(a)), key=lambda i: (-abs(a[i]), i))[:k])
    sb = set(sorted(range(len(b)), key=lambda i: (-abs(b[i]), i))[:k])
    return len(sa & sb) / len(sa | sb)


def topk_with_ties(a, b, k: int = 3, tie_tol: float = 1e-6) -> dict:
    """top-k jaccard, plus a warning when the k-th and (k+1)-th magnitudes are a near tie.

    a rank flip across a near tie says nothing about the explanation.
    """
    out = {"jaccard": topk_overlap(a, b, k), "near_tie": False}
    for v in (a, b):
        mags = np.sort(np.abs(np.asarray(v, float)))[::-1]
        if len(mags) > k and mags[k - 1] - mags[k] <= tie_tol * max(1.0, mags[k - 1]):
            out["near_tie"] = True
    return out


def top_k(values, k: int) -> list[int]:
    v = np.asarray(values, float)
    return sorted(np.argsort(-v, kind="stable")[:k].tolist())
