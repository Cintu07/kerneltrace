"""read evidence/bf16_audit/*.json, print the numbers, and score the written predictions.

usage: python scripts/bf16_summary.py
writes evidence/bf16_audit/summary.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DIR = Path("evidence/bf16_audit")
ORDER = ("gpt2", "pythia-160m", "pythia-410m", "pythia-1b")
SETTINGS = ("fp32", "bf16", "bf16 + fp32 head", "bf16 + fp32 attention and head",
            "bf16, key bias dropped", "bf16, key bias dropped + fp32 head")


def step(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def top3(restore):
    return sorted(np.argsort(-np.nanmean(np.asarray(restore), axis=0))[:3].tolist())


def summarize(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {"model": d["model"], "n_prompts": len(d["prompts"]), "settings": {}}
    ref = d["settings"]["fp32"]["rows"]
    ref_same = np.array([r["restore_same_shape"] for r in ref])
    ref_sweep = np.array([r["restore_sweep_batched"] for r in ref])
    contrast = np.array([r["clean"] - r["corrupt"] for r in ref])
    logit = np.array([abs(r["answer_logit"]) for r in ref])
    out["median_abs_answer_logit"] = float(np.median(logit))
    out["bf16_step_at_answer_logit"] = step(float(np.median(logit)))
    out["median_contrast"] = float(np.median(contrast))
    for label in SETTINGS:
        rows = d["settings"][label]["rows"]
        gap = np.array([abs(r["corrupt"] - r["corrupt_batched"]) for r in rows])
        same = np.array([r["restore_same_shape"] for r in rows])
        sweep = np.array([r["restore_sweep_batched"] for r in rows])
        e_same, e_sweep = np.abs(same - ref_same), np.abs(sweep - ref_sweep)
        ok = ~np.isnan(same).any(1)  # prompts where the setting still saw an effect
        zero = int((~ok).sum())
        margins_seen = np.array([[r["clean"], r["corrupt"]] for r in rows])
        # per prompt: does the single best layer stay the best layer?
        top1_same = float(np.mean(same[ok].argmax(1) == ref_same[ok].argmax(1)))
        top1_sweep = float(np.mean(sweep[ok].argmax(1) == ref_sweep[ok].argmax(1)))
        out["settings"][label] = {
            "batch_gap_median": float(np.median(gap)), "batch_gap_max": float(gap.max()),
            "batch_gap_share_ge_0.25": float(np.mean(gap >= 0.25)),
            "zero_contrast_prompts": zero,
            "distinct_margin_values": int(len(np.unique(margins_seen))),
            "restore_err_same_median": float(np.median(e_same[ok])), "restore_err_same_max": float(e_same[ok].max()),
            "restore_err_sweep_median": float(np.median(e_sweep[ok])), "restore_err_sweep_p95": float(np.quantile(e_sweep[ok], 0.95)),
            "restore_err_sweep_max": float(e_sweep[ok].max()),
            "top3_same": top3(same), "top3_sweep": top3(sweep),
            "top1_per_prompt_agree_same": top1_same, "top1_per_prompt_agree_sweep": top1_sweep,
            "gaps": gap.tolist(),
        }
    return out


def main():
    res = {n: summarize(DIR / f"{n}.json") for n in ORDER if (DIR / f"{n}.json").exists()}
    names = [n for n in ORDER if n in res]
    for n in names:
        r = res[n]
        print(f"\n{n}  ({r['n_prompts']} prompts)  |answer logit| ~{r['median_abs_answer_logit']:.0f}  "
              f"bf16 step there {r['bf16_step_at_answer_logit']}  median contrast {r['median_contrast']:.2f}")
        for label in SETTINGS:
            s = r["settings"][label]
            print(f"  {label:34s} batch gap med {s['batch_gap_median']:.3f} max {s['batch_gap_max']:.3f} "
                  f">=.25 {s['batch_gap_share_ge_0.25']:.0%} | zero effect {s['zero_contrast_prompts']} | sweep err med {s['restore_err_sweep_median']:.3f} "
                  f"p95 {s['restore_err_sweep_p95']:.3f} max {s['restore_err_sweep_max']:.3f} | same-shape err med "
                  f"{s['restore_err_same_median']:.3f} | top3 sweep {s['top3_sweep']} same {s['top3_same']} | "
                  f"top1 agree sweep {s['top1_per_prompt_agree_sweep']:.0%} same {s['top1_per_prompt_agree_same']:.0%}")

    print("\npredictions (experiments/bf16_audit_prediction.txt)")
    S = lambda n, lab: res[n]["settings"][lab]
    p1 = bool(all(S(n, "fp32")["batch_gap_max"] < 1e-3 for n in names))
    p2a = "gpt2" in res and np.mean(np.array(S("gpt2", "bf16")["gaps"]) >= 0.5) >= 0.25
    p2b = all(S(n, "bf16")["batch_gap_median"] > S("gpt2", "bf16")["batch_gap_median"]
              for n in names if n.startswith("pythia")) if "gpt2" in res else False
    p3 = all(S(n, "bf16 + fp32 head")["batch_gap_median"] < 0.1 for n in names)
    p4a = any(S(n, "bf16")["top3_sweep"] != S(n, "fp32")["top3_sweep"] for n in names)
    p4b = all(S(n, "bf16 + fp32 head")["top3_sweep"] == S(n, "fp32")["top3_sweep"] for n in names)
    verdicts = {"1 fp32 gap under 1e-3": p1, "2a gpt-2 bf16 gap >= 0.5 on a quarter": p2a,
                "2b pythia gaps bigger than gpt-2": p2b, "3 fp32 head median gap under 0.1": p3,
                "4a bf16 top-3 differs on some model": p4a, "4b fp32 head top-3 matches everywhere": p4b}
    verdicts = {k: bool(v) for k, v in verdicts.items()}
    for k, v in verdicts.items():
        print(f"  {'held ' if v else 'WRONG'}  {k}")
    (DIR / "summary.json").write_text(json.dumps({"models": res, "predictions": verdicts}, indent=1),
                                      encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
