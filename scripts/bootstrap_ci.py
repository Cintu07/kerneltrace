"""95% bootstrap intervals over prompts for the headline medians.

prompts are the independent unit: each resample draws prompts with replacement and recomputes the pooled
median over all of a prompt's layers (patching error) or pairs (interaction floor).

usage: python scripts/bootstrap_ci.py
writes evidence/bootstrap_ci.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B, SEED = 2000, 0
AUDITS = {"ioi": Path("evidence/bf16_audit"), "capitals": Path("evidence/bf16_audit_capitals"),
          "ioi_sdpa": Path("evidence/bf16_audit_sdpa")}
MODELS = ("gpt2", "pythia-160m", "pythia-410m", "pythia-1b")
FLOOR = Path("evidence/patching_floor")


def ci(per_prompt, rng):
    """per_prompt: list of 1-d arrays (one per prompt). median of the pooled values, and its interval."""
    pooled = np.concatenate(per_prompt)
    stats = []
    n = len(per_prompt)
    for _ in range(B):
        pick = rng.integers(0, n, n)
        stats.append(np.median(np.concatenate([per_prompt[i] for i in pick])))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"median": float(np.median(pooled)), "lo": float(lo), "hi": float(hi), "n_prompts": n}


def audit_errors(d, setting):
    ref = d["settings"]["fp32"]["rows"]
    rows = d["settings"][setting]["rows"]
    out = []
    for r, f in zip(rows, ref):
        a, b = np.array(r["restore_sweep_batched"]), np.array(f["restore_sweep_batched"])
        if np.isnan(a).any():  # the setting saw no effect on this prompt; excluded, as in bf16_summary
            continue
        out.append(np.abs(a - b))
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {"resamples": B, "audit": {}, "floor": {}}
    for task, path in AUDITS.items():
        for m in MODELS:
            f = path / f"{m}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            res["audit"].setdefault(task, {})[m] = {
                s: ci(audit_errors(d, s), rng) for s in d["settings"] if s != "fp32"}
    for m in ("pythia-410m-deduped", "pythia-410m", "gpt2-medium"):
        d = json.loads((FLOOR / f"{m}.json").read_text(encoding="utf-8"))
        runs = d["runs"]
        out = {}
        for drop in ("", ", bias dropped"):
            ref = [np.array(r["q"]) for r in runs["float64" + drop]["rows"]]
            for prec in ("float32", "bfloat16"):
                rows = [np.array(r["q"]) for r in runs[prec + drop]["rows"]]
                out[prec + drop] = ci([np.abs(a - b) for a, b in zip(rows, ref)], rng)
        res["floor"][m] = out
    for task, ms in res["audit"].items():
        for m, ss in ms.items():
            for s, v in ss.items():
                print(f"{task:9s} {m:12s} {s:36s} {v['median']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]  n={v['n_prompts']}")
    for m, ss in res["floor"].items():
        for s, v in ss.items():
            print(f"floor     {m:20s} {s:24s} {v['median']:.2e} [{v['lo']:.2e}, {v['hi']:.2e}]")
    Path("evidence/bootstrap_ci.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
