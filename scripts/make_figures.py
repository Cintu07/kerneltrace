"""Paper-style figures for docs/FINDINGS.md and posts, built only from evidence/ bundles.

Usage: python scripts/make_figures.py   -> docs/figures/*.png and *.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "docs" / "figures"

# Validated categorical slots 1-3, recessive inks, and reserved status colours.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
GOOD, WARN, BAD = "#1a7f37", "#b7791f", "#c62828"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "text.color": INK,
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
})


def bundle(pattern: str) -> dict:
    [path] = [p for p in EVIDENCE.iterdir() if pattern in p.name]
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def panel_label(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 3.3)

    def box(x, y, w, h, title, body, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=edge, lw=1.6))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, fontweight="semibold")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))

    box(0.05, 0.85, 1.8, 1.6, "1. Claim card", "hypothesis\npredictions + thresholds\nfalsification rules\ncommitted before data", INK2)
    box(2.25, 0.85, 1.8, 1.6, "2. Baseline", "patch clean activations\ninto corrupted runs\nheld-out templates\nnull + random controls", INK2)
    box(4.55, 2.15, 2.6, 1.05, "Numerical execution", "kernels, batch size, TF32, dtype\nprofiler checks what actually ran", BLUE)
    box(4.55, 1.05, 2.6, 1.05, "Representation symmetry", "permute every MLP's channels\nfunction unchanged, ids scrambled", ORANGE)
    box(4.55, -0.05, 2.6, 1.05, "Counterexample search", "random / coverage / guided\nshrink the failure, replay it", AQUA)
    box(7.6, 0.85, 2.3, 1.6, "4. Verdict + evidence", "supported / falsified /\nchallenged / inconclusive\nhashed, replayable bundle", INK2)
    ax.text(5.85, 3.27, "3. Try to break it", ha="center", va="bottom", fontsize=10.5, fontweight="semibold")
    arrow(1.85, 1.65, 2.25, 1.65)
    for y in (2.67, 1.57, 0.47):
        arrow(4.05, 1.65, 4.55, y)
        arrow(7.15, y, 7.6, 1.65)
    ax.text(5, -0.38, "A failed prerequisite never becomes a pass: it makes the claim inconclusive.",
            ha="center", fontsize=9, color=INK2, style="italic")
    save(fig, "fig1_pipeline")


def fig_verdicts():
    cards = [
        ("GPT-2 small", "MLP block 0 restores at least half\nof the IOI logit contrast", "SUPPORTED IN SCOPE", GOOD, "✓",
         "held-out restoration 0.84 [0.82, 0.86]\nsurvived every listed challenge"),
        ("GPT-2 small", "late MLP blocks 9-11 carry\nthe IOI decision", "FALSIFIED IN SCOPE", BAD, "✗",
         "held-out restoration 0.084 [0.076, 0.092]\npre-registered threshold: 0.5"),
        ("Pythia-160m", "MLP block 0 restores at least half\nof the IOI logit contrast", "INCONCLUSIVE", WARN, "?",
         "effect replicates: 0.997 [0.985, 1.013]\npermutation parity gate failed"),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    for i, (model, claim, status, colour, icon, detail) in enumerate(cards):
        x = 0.1 + i * 3.32
        ax.add_patch(FancyBboxPatch((x, 0.1), 3.1, 2.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=GRID, lw=1.4))
        ax.text(x + 0.2, 2.6, model, fontsize=9, color=INK2, va="top")
        ax.text(x + 0.2, 2.3, claim, fontsize=10.5, fontweight="semibold", va="top", linespacing=1.3)
        ax.add_patch(FancyBboxPatch((x + 0.2, 1.05), 2.7, 0.45, boxstyle="round,pad=0.01,rounding_size=0.1",
                                    fc=colour, ec="none"))
        ax.text(x + 0.35, 1.275, f"{icon}  {status}", fontsize=9.5, fontweight="bold", color="white", va="center")
        ax.text(x + 0.2, 0.85, detail, fontsize=8.6, color=INK2, va="top", linespacing=1.35)
    save(fig, "fig2_verdicts")


def fig_layers():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (name, pattern), letter in zip(axes, [("GPT-2 small", "gpt2_ioi_mlp0"),
                                                  ("Pythia-160m", "pythia160m_ioi_mlp0_c07")], "ab"):
        rows = bundle(pattern)["layers"]
        for split, colour, dx in (("discovery", BLUE, -0.16), ("holdout", ORANGE, 0.16)):
            rs = [r for r in rows if r["split"] == split]
            x = np.array([r["layer"] for r in rs]) + dx
            m = np.array([r["restoration"]["mean"] for r in rs])
            lo = m - np.array([r["restoration"]["ci95_low"] for r in rs])
            hi = np.array([r["restoration"]["ci95_high"] for r in rs]) - m
            ax.errorbar(x, m, yerr=[lo, hi], fmt="o", ms=5, color=colour, elinewidth=1.4, capsize=0,
                        label=f"{split} templates")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.text(11.4, 0.52, "claim threshold 0.5", ha="right", fontsize=8, color=MUTED)
        ax.set_xticks(range(12))
        ax.set_xlabel("MLP block patched (clean into corrupted run)")
        ax.set_title(name)
        panel_label(ax, letter)
    axes[0].set_ylabel("fraction of logit contrast restored")
    axes[0].legend(loc="upper right")
    fig.suptitle("Restoring MLP block 0 alone recovers most of the IOI answer in both models",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", y=1.04)
    save(fig, "fig3_layers")


def fig_numerical():
    order = ["eager_fp32_repeat", "sdpa_fp32_math", "sdpa_fp32_efficient", "sdpa_fp32_default",
             "eager_fp32_bs1", "eager_fp32_tf32", "eager_fp16", "eager_bf16"]
    labels = {"eager_fp32_repeat": "rerun, identical settings", "sdpa_fp32_math": "SDPA, math kernel",
              "sdpa_fp32_efficient": "SDPA, mem-efficient kernel", "sdpa_fp32_default": "SDPA, default dispatch",
              "eager_fp32_bs1": "batch size 1 (vs 16)", "eager_fp32_tf32": "TF32 matmuls",
              "eager_fp16": "fp16 (out of claim scope)", "eager_bf16": "bf16 (out of claim scope)"}
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(order))
    floor = 1e-6
    for name, pattern, colour, dy in (("GPT-2 small", "gpt2_ioi_mlp0", BLUE, -0.19),
                                      ("Pythia-160m", "pythia160m_ioi_mlp0_c07", ORANGE, 0.19)):
        rows = {r["condition"]: r for r in bundle(pattern)["numerical"]}
        for i, cond in enumerate(order):
            r = rows[cond]
            v = r.get("max_abs_patched_margin_diff")
            broken = v is None or not np.isfinite(v)
            edge = 2e3  # off-scale: NaN has no position, so it is drawn at the axis edge
            v = floor if r.get("bitwise_identical") else (edge if broken else max(v, floor))
            changed = r.get("classification") == "decision_changing"
            if not broken:
                ax.plot([floor, v], [i + dy] * 2, color=colour, lw=2, alpha=0.35, solid_capstyle="round")
            ax.plot(v, i + dy, marker=">" if broken else ("X" if changed else "o"), ms=9 if changed else 7, color=colour,
                    mec="white", mew=1.2, ls="none", label=name if i == 0 else None)
            note = None
            if r.get("bitwise_identical"):
                note = "bitwise identical"
            elif broken:
                note = "NaN on every case"
            elif changed:
                note = "changes which blocks are selected" if not r["prediction_flips"] else "flips claim predictions"
            if note and broken:
                ax.text(v / 1.8, i + dy, note + " (off scale)", va="center", ha="right", fontsize=8, color=colour)
            elif note:
                ax.text(v * 1.6, i + dy, note, va="center", fontsize=8, color=colour)
    ax.axvline(1e-4, color=MUTED, lw=1, ls="--")
    ax.text(1.15e-4, 0.5, "tolerance 1e-4", fontsize=8, color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.6, 3e3)
    ax.set_yticks(y, [labels[c] for c in order])
    ax.invert_yaxis()
    ax.set_xlabel("largest change in any causal measurement vs reference (logits, log scale)")
    ax.legend(loc="upper right", title="X = a decision changed", title_fontsize=8)
    ax.set_title("At fp32, no kernel or batch-size change altered a conclusion; reduced precision did")
    save(fig, "fig4_numerical")


def fig_symmetry():
    n = bundle("gpt2_ioi_mlp0")["symmetry"]["neuron_level"][0]
    so, so_mapped, st = (np.array(n[k]) for k in ("scores_original", "scores_original_mapped", "scores_transformed"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    axes[0].scatter(so_mapped, st, s=7, color=BLUE, alpha=0.7, lw=0)
    axes[0].set_title(f"through the component map: rho = {n['mapped_spearman']:.4f}")
    axes[1].scatter(so, st, s=7, color=ORANGE, alpha=0.7, lw=0)
    axes[1].set_title(f"same neuron ids, no map: rho = {n['naive_spearman']:.3f}")
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("original model: attribution of each MLP neuron")
        panel_label(ax, letter)
    axes[0].set_ylabel("permuted model: attribution")
    fig.suptitle("GPT-2 layer 0, all 3,072 neurons, after a permutation that leaves the model's function unchanged",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", y=1.05)
    fig.text(0.01, -0.06, f"Patching the top {n['top_k']} neurons restores {n['exact_patch_restoration_original']:.4f} "
                          f"in the original, {n['exact_patch_restoration_mapped']:.4f} via the map, "
                          f"and {n['exact_patch_restoration_naive_ids']:.4f} using the raw ids.",
             fontsize=9, color=INK2)
    save(fig, "fig5_symmetry")


def fig_noise_floor():
    data = json.loads((OUT / "noise_floor.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (model, vals), letter in zip(axes, data["models"].items(), "ab"):
        names = ["permuted vs original", "batch 1 vs batch 16", "batches of 8 vs 16"]
        v = [vals[k] for k in names]
        ax.bar(range(3), v, color=[ORANGE, MUTED, MUTED], width=0.6)
        ax.axhline(data["registered_parity_atol"], color=BAD, lw=1.2, ls="--")
        if letter == "a":
            ax.text(2.4, data["registered_parity_atol"], "registered parity tolerance", ha="right", va="bottom",
                    fontsize=8, color=BAD)
        ax.set_xticks(range(3), ["permuted\nmodel", "same model,\nbatch 1", "same model,\nbatches of 8"])
        ax.set_title(f"{model} (logits up to ~{vals['max_abs_logit']})")
        ax.set_ylim(0, max(max(v), data["registered_parity_atol"]) * 1.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        panel_label(ax, letter)
    axes[0].set_ylabel("largest logit change vs batch of 16")
    fig.suptitle("The 'failed' Pythia permutation moves logits no more than changing the batch size does",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", y=1.04)
    save(fig, "fig6_noise_floor")


if __name__ == "__main__":
    fig_pipeline()
    fig_verdicts()
    fig_layers()
    fig_numerical()
    fig_symmetry()
    fig_noise_floor()
