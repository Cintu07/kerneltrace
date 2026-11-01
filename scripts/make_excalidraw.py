"""hand-drawn (excalidraw) versions of the result figures, generated from evidence/.

chart geometry has to come from the data, so these get generated, and every bar
and cell is to scale. the palette is ink, greys and one blue ramp. negative values
get hatching so the figures stay at one colour.

usage: python scripts/make_excalidraw.py   -> docs/figures/x_*.excalidraw
render each one with the excalidraw renderer to get the pngs.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV, OUT = ROOT / "evidence", ROOT / "docs" / "figures"
INK, GREY, RULE, PAPER, BLUE = "#1f2328", "#6e7781", "#d0d7de", "#ffffff", "#0969da"
RAMP = ["#ffffff", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a"]
HAND, CODE = 5, 3
_seed = itertools.count(1000)


def base(kind, x, y, w, h, **kw):
    e = {"type": kind, "id": f"{kind}{next(_seed)}", "x": x, "y": y, "width": w, "height": h, "angle": 0,
         "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
         "strokeStyle": "solid", "roughness": 1, "opacity": 100, "seed": next(_seed), "version": 1,
         "versionNonce": next(_seed), "isDeleted": False, "groupIds": [], "boundElements": None,
         "link": None, "locked": False}
    e.update(kw)
    return e


def rect(x, y, w, h, **kw):
    kw.setdefault("roundness", {"type": 3})
    return base("rectangle", x, y, w, h, **kw)


def ellipse(x, y, w, h, **kw):
    return base("ellipse", x, y, w, h, **kw)


def text(x, y, s, size=18, color=INK, align="left", font=HAND, w=None):
    lines = s.split("\n")
    width = w or max(len(l) for l in lines) * size * 0.56
    return base("text", x, y, width, len(lines) * size * 1.25, text=s, originalText=s, fontSize=size,
                fontFamily=font, textAlign=align, verticalAlign="top", strokeColor=color, containerId=None,
                lineHeight=1.25, strokeWidth=1)


def line(points, color=INK, width=2, dashed=False, arrow=False):
    x0, y0 = points[0]
    rel = [[px - x0, py - y0] for px, py in points]
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    kind = "arrow" if arrow else "line"
    e = base(kind, x0, y0, max(xs) - min(xs), max(ys) - min(ys), strokeColor=color, strokeWidth=width,
             strokeStyle="dashed" if dashed else "solid", points=rel)
    if arrow:
        e.update(startBinding=None, endBinding=None, startArrowhead=None, endArrowhead="arrow")
    return e


def save(name, elements):
    doc = {"type": "excalidraw", "version": 2, "source": "kerneltrace/scripts/make_excalidraw.py",
           "elements": elements, "appState": {"viewBackgroundColor": PAPER, "gridSize": 20}, "files": {}}
    p = OUT / f"{name}.excalidraw"
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8", newline="\n")
    print("wrote", p)


def bundle(pattern):
    [d] = [d for d in EV.iterdir() if d.is_dir() and pattern in d.name]
    return json.loads((d / "metrics.json").read_text(encoding="utf-8"))


def shade(v):
    """sequential blue for restoration in [0, 1]. returns (fill, text colour)."""
    i = min(len(RAMP) - 1, max(0, round(v * (len(RAMP) - 1))))
    return RAMP[i], (PAPER if i >= 5 else INK)


def fnum(v, d=2):
    return ("−" if v < 0 else "") + f"{abs(v):.{d}f}"


# --- verdicts
def verdicts():
    E = [text(40, 30, "three claims, written down before any data. here is what happened.", 30, INK),
         text(42, 78, "gpt-2 small and pythia-160m, indirect object task, held-out prompts", 18, GREY)]
    cards = [("gpt-2 small", "mlp block 0 restores at least\nhalf of the answer", "supported_in_scope",
              "gpt2_ioi_mlp0", "survived all three attacks"),
             ("gpt-2 small", "late mlp blocks 9-11 carry\nthe decision", "falsified_in_scope",
              "gpt2_ioi_late_mlp", "needed 0.5, got far less"),
             ("pythia-160m", "mlp block 0 restores at least\nhalf of the answer", "inconclusive",
              "pythia160m_ioi_mlp0_c07", "effect replicates, but the\nshuffle check failed, so no pass")]
    for i, (model, claim, status, pat, note) in enumerate(cards):
        m = bundle(pat)
        pred = next(p for p in m["verdict"]["predictions"] if p["id"] == "mean_restoration")["stats"]
        x, y, w = 40 + i * 520, 140, 480
        E.append(rect(x, y, w, 400, strokeColor=RULE, strokeWidth=2))
        E.append(text(x + 28, y + 26, model, 18, GREY))
        E.append(text(x + 28, y + 58, claim, 26, INK))
        style = {"supported_in_scope": dict(backgroundColor=BLUE, strokeColor=BLUE),
                 "falsified_in_scope": dict(backgroundColor=RULE, fillStyle="hachure", strokeColor=INK),
                 "inconclusive": dict(strokeStyle="dashed", strokeColor=INK)}[status]
        E.append(rect(x + 28, y + 150, 290, 52, strokeWidth=2.5, **style))
        label = status.replace("_", " ")
        E.append(text(x + 28, y + 162, label, 22, PAPER if status == "supported_in_scope" else INK,
                      align="center", w=290))
        E.append(text(x + 28, y + 228, fnum(pred["mean"]), 64, BLUE if status == "supported_in_scope" else INK))
        E.append(text(x + 28, y + 312, f"held-out restoration, 95% ci {fnum(pred['ci95_low'])} to "
                                       f"{fnum(pred['ci95_high'])}\n{note}", 17, GREY))
    E.append(text(42, 570, "github.com/Cintu07/kerneltrace", 18, GREY, font=CODE))
    save("x_verdicts", E)


# --- causal trace
def trace():
    data = json.loads((EV / "causal_trace" / "gpt2_ioi_trace.json").read_text(encoding="utf-8"))
    P = data["prompts"]
    roles = [("io", "first name"), ("s1", "second name"), ("s2", "repeated name"), ("end", "last word")]

    def mean(kind, r, l):
        return sum(p["float32"]["trace"][kind][l][p["roles"][r]] for p in P) / len(P)

    E = [text(40, 26, "where the answer lives inside gpt-2", 34, INK),
         text(42, 76, f"each cell restores one activation at one word and one layer. darker = more of the "
                      f"answer comes back. mean of {len(P)} prompts.", 18, GREY),
         text(42, 102, "“when mary and john went to the store, john gave a drink to” -> mary", 18,
              GREY, font=CODE)]
    cw, ch, gx, top = 40, 44, 175, 215
    kinds = [("resid", "residual stream"), ("mlp", "mlp output"), ("attn", "attention output")]
    origin = {}
    for k, (kind, title) in enumerate(kinds):
        x0 = gx + k * (12 * cw + 70)
        origin[kind] = x0
        E.append(text(x0, top - 64, title, 22, INK))
        for l in range(0, 12, 2):  # layer numbers above the grid, so annotation arrows from below cross nothing
            E.append(text(x0 + l * cw, top - 26, str(l), 15, GREY, align="center", w=cw - 3))
        if k == 0:
            E.append(text(36, top - 26, "layer", 15, GREY, align="right", w=gx - 50))
        for i, (r, name) in enumerate(roles):
            if k == 0:
                E.append(text(36, top + i * ch + 11, name, 18, INK if r in ("s2", "end") else GREY,
                              align="right", w=gx - 50))
            for l in range(12):
                v = mean(kind, r, l)
                x, y = x0 + l * cw, top + i * ch
                if v <= -0.05:
                    E.append(rect(x, y, cw - 3, ch - 3, backgroundColor=RULE, fillStyle="hachure",
                                  strokeColor=GREY, strokeWidth=1, roundness=None))
                else:
                    fill, tc = shade(max(v, 0))
                    E.append(rect(x, y, cw - 3, ch - 3, backgroundColor=fill,
                                  strokeColor=fill if fill != PAPER else RULE, strokeWidth=1, roundness=None))
                if v >= 0.095 or v <= -0.05:
                    _, tc = shade(max(v, 0))
                    s = "1" if v > 0.995 else fnum(v).replace("0.", ".")
                    E.append(text(x, y + 12, s, 15, tc if v > 0 else INK, align="center", font=CODE, w=cw - 3))
    # annotations, pointing at the cells they describe
    ann_y = top + 4 * ch + 70
    rx, mx, ax = origin["resid"], origin["mlp"], origin["attn"]
    E.append(line([(rx + 3.5 * cw, ann_y - 6), (rx + 3.5 * cw, top + 3 * ch - 4)], GREY, 1.5, arrow=True))
    E.append(text(rx - 20, ann_y, "the answer sits on the repeated\nname for layers 0-6, then moves\nto the last word at layers 7-8", 18, INK))
    E.append(line([(mx + 0.5 * cw, ann_y - 6), (mx + 0.5 * cw, top + 3 * ch - 4)], GREY, 1.5, arrow=True))
    E.append(text(mx - 10, ann_y, "only mlp 0 matters, and only\non the repeated name (0.85).\nthat is the claim that held", 18, INK))
    E.append(line([(ax + 11.5 * cw, ann_y - 6), (ax + 11.5 * cw, top + 4 * ch - 4)], GREY, 1.5, arrow=True))
    E.append(line([(ax + 8 * cw, ann_y - 6), (ax + 8 * cw, top + 4 * ch - 4)], GREY, 1.5, arrow=True))
    E.append(text(ax + 40, ann_y, "layers 7-9 carry the answer to the\nend. layers 10-11 push back against\nit (hatched = negative)", 18, INK))
    E.append(text(42, ann_y + 110, "same layer pattern as the published ioi circuit, found from scratch. "
                                   "layer level, not individual heads.", 17, GREY))
    save("x_trace", E)


# --- numerical attacks
def numerical():
    order = [("eager_fp32_repeat", "rerun, same settings"), ("sdpa_fp32_math", "different attention kernel (math)"),
             ("sdpa_fp32_efficient", "different attention kernel (mem-efficient)"),
             ("eager_fp32_bs1", "batch size 1 instead of 16"), ("eager_fp32_tf32", "tf32 matmuls"),
             ("eager_fp16", "fp16 (outside the claim)"), ("eager_bf16", "bf16 (outside the claim)")]
    models = [("gpt-2 small", "gpt2_ioi_mlp0", INK, -9), ("pythia-160m", "pythia160m_ioi_mlp0_c07", BLUE, 9)]
    rows = {name: {r["condition"]: r for r in bundle(pat)["numerical"]} for name, pat, _, _ in models}
    E = [text(40, 26, "attack 1: run it differently, see if the explanation moves", 32, INK),
         text(42, 74, "largest change in any causal measurement vs the reference run, in logits (log scale)", 18, GREY)]
    L, R, top, rh = 500, 1500, 150, 62
    lo, hi = -7, 2

    def X(v):
        return L + (R - L) * (math.log10(v) - lo) / (hi - lo)

    for e in range(lo, hi + 1):
        E.append(line([(X(10 ** e), top - 20), (X(10 ** e), top + len(order) * rh)], RULE, 1))
        E.append(text(X(10 ** e) - 30, top + len(order) * rh + 10, f"1e{e}", 15, GREY, align="center", font=CODE, w=60))
    E.append(line([(X(1e-4), top - 30), (X(1e-4), top + len(order) * rh)], GREY, 2, dashed=True))
    E.append(text(X(1e-4) + 8, top - 40, "tolerance", 16, GREY))
    for i, (cond, label) in enumerate(order):
        yc = top + i * rh + rh / 2
        E.append(text(40, yc - 12, label, 19, INK, align="right", w=L - 70))
        for name, _, col, dy in models:
            r = rows[name][cond]
            y = yc + dy
            if r.get("bitwise_identical"):
                E.append(text(X(1e-7) + 4, y - 10, "bitwise identical", 15, col))
                continue
            v = r.get("max_abs_patched_margin_diff")
            if v is None or not math.isfinite(v):
                E.append(text(X(10 ** hi) - 210, y - 10, "nan on every prompt ->", 15, col))
                continue
            E.append(line([(X(1e-7), y), (X(v), y)], col, 2))
            if r.get("classification") == "decision_changing":
                d = 9
                E.append(line([(X(v) - d, y - d), (X(v) + d, y + d)], col, 3.5))
                E.append(line([(X(v) - d, y + d), (X(v) + d, y - d)], col, 3.5))
            else:
                E.append(ellipse(X(v) - 7, y - 7, 14, 14, backgroundColor=col, strokeColor=col))
    ly = top + len(order) * rh + 60
    E.append(line([(L, ly + 8), (L + 40, ly + 8)], INK, 2))
    E.append(text(L + 50, ly - 4, "gpt-2 small", 17, INK))
    E.append(line([(L + 200, ly + 8), (L + 240, ly + 8)], BLUE, 2))
    E.append(text(L + 250, ly - 4, "pythia-160m", 17, BLUE))
    E.append(text(L + 420, ly - 4, "x = a conclusion changed", 17, GREY))
    E.append(text(40, ly + 40, "at full precision nothing changed a conclusion. bf16 changed which blocks gpt-2 ranks top three "
                               "and flipped pythia's predictions.", 17, GREY))
    save("x_numerical", E)


# --- neuron shuffle
def shuffle():
    n = bundle("gpt2_ioi_mlp0")["symmetry"]["neuron_level"][0]
    E = [text(40, 26, "attack 2: shuffle the neurons, keep the math identical", 32, INK),
         text(42, 74, "every mlp's 3,072 neurons reordered, with the inverse order on the weights that read them. "
                      "same outputs.", 18, GREY)]
    ids_a = [12, 87, 340, 901, 1544, 2210, 2873]
    perm = [4, 0, 6, 2, 5, 1, 3]
    xa, xb, y0, gap = 120, 460, 170, 58
    E.append(text(xa - 40, y0 - 44, "original", 18, GREY))
    E.append(text(xb - 30, y0 - 44, "shuffled", 18, GREY))
    for i, nid in enumerate(ids_a):
        y = y0 + i * gap
        E.append(ellipse(xa, y, 26, 26, backgroundColor=BLUE if i == 2 else PAPER, strokeColor=INK))
        E.append(text(xa - 90, y + 2, f"#{nid}", 16, GREY, align="right", w=80, font=CODE))
        j = perm[i]
        yb = y0 + j * gap
        E.append(line([(xa + 30, y + 13), (xb - 4, yb + 13)], BLUE if i == 2 else RULE, 2 if i == 2 else 1.5))
    for j in range(len(ids_a)):
        y = y0 + j * gap
        src = perm.index(j)
        E.append(ellipse(xb, y, 26, 26, backgroundColor=BLUE if src == 2 else PAPER, strokeColor=INK))
        E.append(text(xb + 36, y + 2, f"#{ids_a[j]}", 16, GREY, font=CODE))
    sx = 680
    E.append(text(sx, 160, "read through the map", 22, INK))
    E.append(text(sx, 196, f"rho {n['mapped_spearman']:.4f}", 56, BLUE, font=CODE))
    E.append(text(sx, 270, f"top-20 neurons restore {n['exact_patch_restoration_mapped']:.4f},\n"
                           f"same as the original ({n['exact_patch_restoration_original']:.4f})", 18, GREY))
    E.append(text(sx + 520, 160, "read by raw neuron number", 22, INK))
    E.append(text(sx + 520, 196, f"rho {fnum(n['naive_spearman'], 3)}", 56, INK, font=CODE))
    E.append(text(sx + 520, 270, f"the same top-20 ids now restore\n{fnum(n['exact_patch_restoration_naive_ids'], 4)}: "
                                 f"they point at the wrong neurons", 18, GREY))
    E.append(text(sx, 380, "a claim like “neuron 340 does x” only survives a harmless relabelling\n"
                           "if you carry the map. gpt-2 layer 0, all 3,072 neurons.", 20, INK))
    save("x_shuffle", E)


# --- pythia noise floor
def noise():
    d = json.loads((OUT / "noise_floor.json").read_text(encoding="utf-8"))
    E = [text(40, 26, "the claim that came back inconclusive, and why", 32, INK),
         text(42, 74, "largest logit change, measured over the full vocabulary on 16 prompts", 18, GREY)]
    base_y, H, scale = 460, 330, 330 / 6e-3
    groups = [("gpt-2 small", d["models"]["GPT-2 small"], 120), ("pythia-160m", d["models"]["Pythia-160m"], 760)]
    keys = [("permuted vs original", "shuffled\nmodel"), ("batch 1 vs batch 16", "same model,\nbatch 1"),
            ("batches of 8 vs 16", "same model,\nbatches of 8")]
    tol_y = base_y - d["registered_parity_atol"] * scale
    E.append(line([(80, tol_y), (1420, tol_y)], INK, 2, dashed=True))
    E.append(text(1430, tol_y - 12, "the check's\ntolerance", 16, INK))
    for name, vals, gx in groups:
        E.append(text(gx, base_y + 70, name, 22, INK))
        for i, (k, lab) in enumerate(keys):
            v = vals[k]
            x = gx + i * 190
            h = max(v * scale, 3)
            E.append(rect(x, base_y - h, 120, h, backgroundColor=BLUE if i == 0 else RULE,
                          strokeColor=BLUE if i == 0 else GREY, roundness=None,
                          fillStyle="solid" if i == 0 else "hachure"))
            E.append(text(x, base_y - h - 30, f"{v:.1e}", 16, INK, align="center", w=120, font=CODE))
            E.append(text(x, base_y + 10, lab, 16, GREY, align="center", w=120))
        E.append(line([(gx - 10, base_y), (gx + 2 * 190 + 130, base_y)], INK, 2))
    E.append(text(42, base_y + 120, "pythia's check failed because its logits are 5x bigger, so ordinary float rounding is bigger. "
                                    "running the same model at a\ndifferent batch size moves them just as much. the shuffle "
                                    "was exact and the check was wrong. the claim stays inconclusive anyway,\nrather than "
                                    "tuning the tolerance until it passes.", 18, INK))
    save("x_noise", E)


# --- the readme diagram: tall enough to read at github's column width
def method():
    W = 1100
    E = [text(0, 20, "kerneltrace", 46, INK, align="center", w=W),
         text(0, 84, "write the claim down, then try to break it", 24, GREY, align="center", w=W)]

    def step(y, n, title):
        E.append(text(60, y, f"{n}", 30, BLUE))
        E.append(text(100, y + 2, title, 28, INK))

    def down(x, y0, y1):
        E.append(line([(x, y0), (x, y1)], INK, 2.5, arrow=True))

    # 1. the claim, as it actually sits in git
    step(150, 1, "write the claim down first")
    E.append(rect(100, 200, 950, 130, backgroundColor=INK, strokeColor=INK))
    E.append(text(130, 222, "claim: mlp block 0 restores at least half the answer\nthreshold: 0.5 on prompts it has never seen\ncommitted to git before any measurement",
                  22, PAPER, font=CODE))
    down(W / 2, 362, 412)

    # 2. the measurement
    step(420, 2, "measure it on unseen prompts")
    E.append(text(100, 470, "0.84", 72, BLUE))
    E.append(text(300, 482, "of the answer comes back on 129 held-out\nprompts (95% ci 0.82 to 0.86)", 22, GREY))
    down(W / 2, 580, 630)

    # 3. three attacks side by side, converging on the verdict
    step(640, 3, "try to break it")
    attacks = [("run it differently", "other attention kernels,\nbatch sizes, tf32, bf16"),
               ("shuffle the neurons", "same math, every\nneuron id scrambled"),
               ("search for a breaker", "thousands of prompts,\nshrink any that break it")]
    bw, gap, top = 300, 50, 692
    for i, (head, body) in enumerate(attacks):
        x = 100 + i * (bw + gap)
        E.append(rect(x, top, bw, 150, strokeColor=INK, strokeWidth=2.5))
        E.append(text(x + 22, top + 22, head, 24, INK))
        E.append(text(x + 22, top + 66, body, 20, GREY))
        E.append(line([(x + bw / 2, top + 156), (W / 2, top + 222)], INK, 2, arrow=True))

    # 4. the three possible outcomes
    step(924, 4, "verdict")
    pills = [("supported", dict(backgroundColor=BLUE, strokeColor=BLUE), PAPER),
             ("falsified", dict(backgroundColor=RULE, fillStyle="hachure", strokeColor=INK), INK),
             ("inconclusive", dict(strokeStyle="dashed", strokeColor=INK), INK)]
    for i, (label, style, tc) in enumerate(pills):
        x = 100 + i * (bw + gap)
        E.append(rect(x, 976, bw, 64, strokeWidth=2.5, **style))
        E.append(text(x, 992, label, 26, tc, align="center", w=bw))
    E.append(text(0, 1072, "a failed check never turns into a pass. it turns into inconclusive.", 22, GREY,
                  align="center", w=W))
    save("x_method", E)


if __name__ == "__main__":
    method()
    verdicts()
    trace()
    numerical()
    shuffle()
    noise()
