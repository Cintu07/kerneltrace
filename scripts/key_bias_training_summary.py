"""summarise the key bias training runs and score experiments/key_bias_training_prediction.txt.

usage: python scripts/key_bias_training_summary.py
reads evidence/key_bias_training/*.json, writes evidence/key_bias_training/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIR = Path("evidence/key_bias_training")


def load() -> dict:
    runs = {}
    for p in sorted(DIR.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        runs[d["run"]] = d
    return runs


def final(run: dict) -> dict:
    rows = [r for r in run["log"] if "invariant" in r]
    last = rows[-1]
    ratios = [gi / gr for r in rows[1:] for gi, gr in zip(r["grad_invariant"], r["grad_rotary"]) if gr > 0]
    steps = [si / sr for r in rows[1:] for si, sr in zip(r["step_invariant"], r["step_rotary"]) if sr > 0]
    vals = [r["val_loss"] for r in rows if "val_loss" in r]
    return {"step": last["step"], "diverged": any(r.get("diverged") for r in run["log"]),
            "invariant": max(last["invariant"]), "rotary": max(last["rotary"]),
            "invariant_over_rotary": max(i / r for i, r in zip(last["invariant"], last["rotary"]) if r > 0),
            "query": max(last["query"]),
            "median_grad_ratio": float(np.median(ratios)) if ratios else float("nan"),
            "median_step_ratio": float(np.median(steps)) if steps else float("nan"),
            "val_loss": vals[-1] if vals else float("nan")}


def main():
    runs = {k: final(v) for k, v in load().items()}
    print(f"{'run':20s} {'step':>5s} {'invariant':>10s} {'rotary':>8s} {'inv/rot':>8s} {'grad i/r':>9s} "
          f"{'step i/r':>9s} {'val loss':>9s}")
    for k, f in runs.items():
        print(f"{k:20s} {f['step']:5d} {f['invariant']:10.3f} {f['rotary']:8.3f} {f['invariant_over_rotary']:8.2f} "
              f"{f['median_grad_ratio']:9.2e} {f['median_step_ratio']:9.3f} {f['val_loss']:9.3f}")
    g = runs.get
    checks = []

    def check(n, text, ok):
        checks.append({"n": n, "prediction": text, "held": None if ok is None else bool(ok)})

    have = lambda *names: all(n in runs for n in names)
    check(1, "fp32 adam invariant < 1.0", have("fp32 adam") and g("fp32 adam")["invariant"] < 1.0)
    check(2, "fp16 adam invariant > 2x rotary in some layer",
          have("fp16 adam") and g("fp16 adam")["invariant_over_rotary"] > 2)
    check(3, "fp16 adam invariant >= 5x fp32 adam",
          have("fp16 adam", "fp32 adam") and g("fp16 adam")["invariant"] >= 5 * g("fp32 adam")["invariant"])
    check(4, "bf16 adam invariant >= fp16 adam",
          have("bf16 adam", "fp16 adam") and g("bf16 adam")["invariant"] >= g("fp16 adam")["invariant"])
    check(5, "fp16 adam eps 1e-4 invariant <= half of fp16 adam",
          have("fp16 adam eps 1e-4", "fp16 adam") and g("fp16 adam eps 1e-4")["invariant"] <= 0.5 * g("fp16 adam")["invariant"])
    check(6, "fp16 sgd invariant < 1.0", have("fp16 sgd") and g("fp16 sgd")["invariant"] < 1.0)
    check(7, "grad ratio < 1e-3 in fp32 adam and > 1e-2 in fp16 adam",
          have("fp32 adam", "fp16 adam") and g("fp32 adam")["median_grad_ratio"] < 1e-3 and g("fp16 adam")["median_grad_ratio"] > 1e-2)
    check(8, "fp16 adam val loss within 2% of fp32 adam",
          have("fp16 adam", "fp32 adam") and abs(g("fp16 adam")["val_loss"] / g("fp32 adam")["val_loss"] - 1) < 0.02)
    check(9, "fp16 adam fp32 attention invariant < 0.04",
          have("fp16 adam fp32 attention") and g("fp16 adam fp32 attention")["invariant"] < 0.04)
    print()
    for c in checks:
        print(f"{c['n']}. {'held' if c['held'] else 'WRONG'}  {c['prediction']}")
    print(f"\n{sum(c['held'] for c in checks)} of {len(checks)} held")
    (DIR / "summary.json").write_text(json.dumps({"runs": runs, "predictions": checks}, indent=1),
                                      encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
