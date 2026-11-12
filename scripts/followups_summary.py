"""score experiments/followups_prediction.txt (1-16) against the evidence on disk.

usage: python scripts/followups_summary.py
writes evidence/followups_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bf16_summary import summarize  # noqa: E402

FAM = Path("evidence/key_bias_families")
FLOOR = Path("evidence/patching_floor/summary.json")
CAP = Path("evidence/bf16_audit_capitals")


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    v, notes = {}, {}
    off = load(FAM / "score_offsets.json") or {}
    if off:
        sc = off["santacoder"]
        v["1 santacoder q.k over fp16 max, under it with the bias dropped"] = (
            sc["as released"]["max_matmul"] > 65504 and sc["bias dropped"]["max_matmul"] < 65504)
        v["2 santacoder bf16 damage over 0.1 on under 1% of rows"] = sc["as released"]["share_tv_bf16_over_0.1"] < 0.01
        p = off["pythia-160m"]
        a, b = p["as released"]["share_tv_bf16_over_0.1"], p["bias dropped"]["share_tv_bf16_over_0.1"]
        v["3 pythia-160m damage share over 1%, cut 10x by the drop"] = a > 0.01 and b * 10 <= a
        bl = off["bloom-560m"]
        v["4 bloom-560m drop cuts its largest offset by less than half"] = (
            bl["bias dropped"]["largest_offset"] > 0.5 * bl["as released"]["largest_offset"])
        q = off["qwen2.5-0.5b"]
        v["5 qwen2.5-0.5b drop cuts its largest offset by at least half"] = (
            q["bias dropped"]["largest_offset"] <= 0.5 * q["as released"]["largest_offset"])
        notes["offsets"] = {n: {k: {x: r[k][x] for x in ("largest_offset", "spread_there", "max_matmul",
                                                          "share_tv_bf16_over_0.1", "tv_bf16_p99")}
                                for k in ("as released", "bias dropped")} for n, r in off.items()}
    fl = load(FLOOR) or {}
    if "pythia-410m-deduped" in fl:
        d = fl["pythia-410m-deduped"]
        raw = load(Path("evidence/patching_floor/pythia-410m-deduped.json"))
        removed = raw["runs"]["float64, bias dropped"]["removed"]
        notes["pythia-410m-deduped invariant key bias"] = removed
        v["6 pythia-410m-deduped invariant key bias over 50"] = removed > 50
        v["7 pythia-410m-deduped fp32 floor falls at least 3x with the drop"] = (
            d["float32, bias dropped"]["floor"] * 3 <= d["float32"]["floor"])
        if "gpt2-medium" in fl:
            g = fl["gpt2-medium"]
            v["8 after the drop, within 3x of gpt-2-medium's floor"] = (
                d["float32, bias dropped"]["floor"] <= 3 * g["float32"]["floor"])
            r = g["float32, bias dropped"]["floor"] / g["float32"]["floor"]
            v["9 gpt-2-medium floor moves under 1.5x with the drop"] = 1 / 1.5 < r < 1.5
        v["10 fp64 with and without the bias agree to 1e-9"] = all(
            m["fp64_bias_vs_dropped_max"] < 1e-9 for m in fl.values())
        a, b = d["bfloat16"]["share_err_over_signal"], d["bfloat16, bias dropped"]["share_err_over_signal"]
        v["11 bf16 error over signal on >half the pairs, halved by the drop"] = a > 0.5 and b <= a / 2
        notes["floor"] = fl
    caps = {n: summarize(CAP / f"{n}.json") for n in ("gpt2", "pythia-160m", "pythia-410m", "pythia-1b")
            if (CAP / f"{n}.json").exists()}
    S = lambda n, lab: caps[n]["settings"][lab]
    if "pythia-410m" in caps:
        v["12 pythia-410m capitals bf16 err >= 0.05, under 0.02 with the drop"] = (
            S("pythia-410m", "bf16")["restore_err_sweep_median"] >= 0.05
            and S("pythia-410m", "bf16, key bias dropped")["restore_err_sweep_median"] < 0.02)
    if "pythia-160m" in caps:
        v["13 pythia-160m capitals bf16 top-3 differs, fp32 attention+head matches"] = (
            S("pythia-160m", "bf16")["top3_sweep"] != S("pythia-160m", "fp32")["top3_sweep"]
            and S("pythia-160m", "bf16 + fp32 attention and head")["top3_sweep"] == S("pythia-160m", "fp32")["top3_sweep"])
        v["15 pythia-160m capitals fp32 head worse than bf16"] = (
            S("pythia-160m", "bf16 + fp32 head")["restore_err_sweep_median"] > S("pythia-160m", "bf16")["restore_err_sweep_median"])
    if "pythia-1b" in caps and "gpt2" in caps:
        v["14 pythia-1b and gpt-2 capitals bf16 err under 0.03"] = all(
            S(n, "bf16")["restore_err_sweep_median"] < 0.03 for n in ("pythia-1b", "gpt2"))
    ba = load(FAM / "bloom_alibi.json") or {}
    if "bloom-560m" in ba:
        v["16 bloom-560m relative alibi brings bf16 within 2%"] = abs(
            ba["bloom-560m"]["settings"]["bf16, relative alibi"]["vs_fp32"]) < 0.02
    for n, c in caps.items():
        print(f"\ncapitals {n} ({c['n_prompts']} prompts, median contrast {c['median_contrast']:.2f})")
        for lab, s in c["settings"].items():
            print(f"  {lab:36s} sweep err med {s['restore_err_sweep_median']:.3f} p95 {s['restore_err_sweep_p95']:.3f} "
                  f"max {s['restore_err_sweep_max']:.3f} | zero {s['zero_contrast_prompts']} | top3 {s['top3_sweep']} | "
                  f"top1 agree {s['top1_per_prompt_agree_sweep']:.0%}")
    v = {k: bool(x) for k, x in sorted(v.items(), key=lambda kv: int(kv[0].split()[0]))}
    print("\npredictions (experiments/followups_prediction.txt)")
    for k, x in v.items():
        print(f"  {'held ' if x else 'WRONG'}  {k}")
    print(f"  {sum(v.values())} of {len(v)} held")
    Path("evidence/followups_summary.json").write_text(
        json.dumps({"predictions": v, "capitals": caps, **notes}, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
