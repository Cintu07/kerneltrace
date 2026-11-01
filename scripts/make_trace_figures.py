"""paper-style causal-trace figures from evidence/causal_trace/gpt2_ioi_trace.json.

usage: python scripts/make_trace_figures.py -> docs/figures/fig7_*.png, fig8_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence" / "causal_trace" / "gpt2_ioi_trace.json"
OUT = ROOT / "docs" / "figures"

INK, INK2, GRID = "#16181d", "#555b66", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# diverging blue <-> red through a neutral grey. blue brings the answer back, red pushes against it.
# the stops sit on the normalised scale so exactly 0 lands on the neutral grey.
CMAP = LinearSegmentedColormap.from_list("restoration", [
    (0.0, "#a32020"), (0.25, "#e7a59d"), (0.5, "#f0efec"), (0.6, "#cde2fb"),
    (0.72, "#6da7ec"), (0.85, "#256abf"), (1.0, "#0d366b")])
NORM = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=1.0)
KIND_TITLE = {"resid": "residual stream", "mlp": "MLP output", "attn": "attention output"}
ROLES = [("io", "first name (IO)"), ("s1", "second name (S1)"), ("s2", "repeated name (S2)"), ("end", "final token")]

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 9.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "text.color": INK, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def letter(ax, s, dx=-0.02):
    ax.text(dx, 1.0, s, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="right", va="bottom")


def load():
    return json.loads(DATA.read_text(encoding="utf-8"))


def role_matrix(prompts, kind, prec="float32"):
    """[prompt, role, layer]"""
    return np.array([[[p[prec]["trace"][kind][l][p["roles"][r]] for l in range(12)] for r, _ in ROLES]
                     for p in prompts], dtype=float)


def typical_prompt(prompts):
    """the prompt whose role-aligned residual trace sits closest to the mean (the caption says so)."""
    m = role_matrix(prompts, "resid")
    d = ((m - m.mean(0)) ** 2).sum(axis=(1, 2))
    return prompts[int(np.argmin(d))]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def fig_example(data):
    p = typical_prompt(data["prompts"])
    toks = [t.replace(" ", "␣", 1) if t.startswith(" ") else t for t in p["tokens"]]
    toks = [t.replace("␣", " ") for t in toks]
    r = p["roles"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, kind, let in zip(axes, ("resid", "mlp", "attn"), "abc"):
        grid = np.array(p["float32"]["trace"][kind]).T  # [pos, layer]
        im = ax.imshow(grid, cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest")
        ax.set_title(KIND_TITLE[kind])
        ax.set_xticks(range(0, 12, 2))
        ax.set_xlabel("layer")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(toks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        letter(ax, let)
    axes[0].set_yticks(range(len(toks)))
    labels = []
    for i, t in enumerate(toks):
        tag = {r["s2"]: "  S2*", r["io"]: "  IO", r["s1"]: "  S1", r["end"]: "  end"}.get(i, "")
        labels.append(f"{t.strip()}{tag}")
    axes[0].set_yticklabels(labels)
    for i, lab in enumerate(axes[0].get_yticklabels()):
        if i == r["s2"] or i == r["end"]:
            lab.set_fontweight("bold")
            lab.set_color(INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, ticks=[-0.4, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.set_label("fraction of the answer restored", color=INK2)
    fig.suptitle("Restoring one (token, layer) site of GPT-2 small at a time", x=0.125, ha="left",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.text(0.125, 0.985, f"clean: “{p['clean']}”  → {p['answer']}      "
             f"corrupted: S2 changed to {p['answer']}  → {p['distractor']}",
             fontsize=9, color=INK2, ha="left")
    fig.text(0.125, -0.06,
             "Each cell patches the clean activation at one token and one layer into the corrupted run. "
             "1 = the clean answer fully comes back, 0 = no effect,\nnegative = pushes further toward the wrong name. "
             f"Shown: the most typical of {len(data['prompts'])} prompts (closest to the mean residual trace). "
             "fp32, eager attention, pinned weights.", fontsize=8.5, color=INK2, ha="left")
    save(fig, "fig7_causal_trace_example")


def fig_summary(data):
    P = data["prompts"]
    fig = plt.figure(figsize=(10.2, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.62, wspace=0.12)
    for j, kind in enumerate(("resid", "mlp", "attn")):
        ax = fig.add_subplot(gs[0, j])
        m = role_matrix(P, kind).mean(0)
        ax.imshow(m, cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest")
        for i in range(m.shape[0]):
            for l in range(12):
                v = m[i, l]
                if abs(v) >= 0.05:
                    ax.text(l, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center",
                            va="center", fontsize=6.8, color="white" if v > 0.55 or v < -0.3 else INK)
        ax.set_title(KIND_TITLE[kind])
        ax.set_xticks(range(0, 12, 2))
        ax.set_xlabel("layer")
        ax.set_yticks(range(len(ROLES)))
        ax.set_yticklabels([lab for _, lab in ROLES] if j == 0 else [])
        if j:
            ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        if j == 0:
            letter(ax, "a", dx=-0.45)

    # b: the hand-off from the repeated name to the final token
    ax = fig.add_subplot(gs[1, :2])
    rm = role_matrix(P, "resid")
    rng = np.random.default_rng(0)
    for (role, name), col in ((("s2", "repeated name (S2)"), BLUE), (("end", "final token"), ORANGE)):
        v = rm[:, [r for r, _ in ROLES].index(role), :]
        boot = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(2000)])
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        x = np.arange(12)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, v.mean(0), color=col, lw=2, marker="o", ms=4.5)
        if role == "s2":
            ax.text(0.1, 0.9, name, color=col, va="top", fontsize=9.5, fontweight="bold")
        else:
            ax.text(10.9, 0.9, name, color=col, va="top", ha="right", fontsize=9.5, fontweight="bold")
    ax.set_xlim(-0.3, 11.3)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xticks(range(12))
    ax.set_xlabel("layer")
    ax.set_ylabel("restored by patching the\nresidual stream at that token")
    ax.axvspan(6.5, 8.5, color=GRID, alpha=0.6, lw=0)
    ax.text(7.5, 1.1, "hand-off, layers 7–8", ha="center", va="top", fontsize=8.5, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"Where the answer lives, by layer (mean of {len(P)} prompts, 95% CI)")
    letter(ax, "b", dx=-0.08)

    # c: what bf16 does to the same 26k measurements
    ax = fig.add_subplot(gs[1, 2])
    f32 = np.concatenate([np.ravel(np.array(p["float32"]["trace"][k], float)) for p in P for k in ("resid", "mlp", "attn")])
    b16 = np.concatenate([np.ravel(np.array(p["bfloat16"]["trace"][k], float)) for p in P for k in ("resid", "mlp", "attn")])
    d = np.abs(b16 - f32)
    d = d[np.isfinite(d)]
    bins = np.logspace(-6, 0, 37)
    ax.hist(np.clip(d, 1e-6, None), bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(0.05, color=INK2, lw=1, ls="--")
    big = int((d >= 0.05).sum())
    ax.text(2e-4, ax.get_ylim()[1] * 0.97, f"{big:,} cells ({big / d.size:.0%})\nmove by ≥ 0.05",
            fontsize=8.5, color=INK2, va="top", ha="center")
    ax.set_xlabel("|bf16 − fp32| per cell (≤ 1e-6 pooled left)")
    ax.set_ylabel("cells")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"bf16 vs fp32, {d.size:,} cells")
    ax.text(0.0, -0.3, f"median {np.median(d):.1e}, max {d.max():.2f}. bf16 rounds logits near\n"
            "100 to steps of 0.5, so sites that do nothing\nstill show up to ±0.3 restoration.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
    letter(ax, "c", dx=-0.16)
    fig.suptitle("GPT-2 small on IOI: the trace averaged over token roles", x=0.125, ha="left",
                 fontsize=12, fontweight="bold", y=0.985)
    save(fig, "fig8_trace_summary")


if __name__ == "__main__":
    d = load()
    fig_example(d)
    fig_summary(d)
