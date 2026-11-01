"""Human-readable report: Markdown, static HTML and SVG figures from metrics.json.

The report never states more than the verdict does. Its first section after
the verdict is the smallest reproducible input that contradicts the claim,
when one was found, because that is the most useful thing a reader can see.
"""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

# Reference categorical slots 1-3 (validated all-pairs), plus recessive inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "svg.fonttype": "none",
        "text.color": INK, "legend.frameon": False,
    })
    return plt


def fig_layers(bundle, metrics) -> str | None:
    rows = metrics.get("layers") or []
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for split, color, dx in (("discovery", BLUE, -0.15), ("holdout", ORANGE, 0.15)):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        x = np.array([r["layer"] for r in rs]) + dx
        m = np.array([r["restoration"]["mean"] for r in rs])
        lo = m - np.array([r["restoration"]["ci95_low"] for r in rs])
        hi = np.array([r["restoration"]["ci95_high"] for r in rs]) - m
        ax.errorbar(x, m, yerr=[lo, hi], fmt="o", ms=5, color=color, ecolor=color,
                    elinewidth=1.5, capsize=0, label=split)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("MLP block patched (clean into corrupt)")
    ax.set_ylabel("restoration of logit contrast")
    ax.set_xticks(sorted({r["layer"] for r in rows}))
    ax.legend(loc="upper right")
    ax.set_title("Single-block restoration, mean with 95% template-cluster bootstrap CI",
                 fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(bundle.path("figures/layer_restoration.svg"))
    plt.close(fig)
    return "figures/layer_restoration.svg"


def fig_numerical(bundle, metrics) -> str | None:
    rows = [r for r in metrics.get("numerical") or [] if r.get("valid")]
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.2))
    names = [f"{r['condition']} ({'+'.join(r['factors'])})" for r in rows]
    vals = [max(r["max_abs_patched_margin_diff"], 1e-12) for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=BLUE, height=0.55)
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for yi, (v, r) in enumerate(zip(vals, rows)):
        label = "bitwise identical" if r["bitwise_identical"] else f"{v:.2g}"
        ax.text(v * 1.3, yi, label, va="center", fontsize=9, color=INK2)
    ax.set_xlabel("max |patched margin difference| vs reference (logits, log scale)")
    ax.set_title("How far each execution condition moves the causal measurements",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlim(right=max(vals) * 30)
    fig.tight_layout()
    fig.savefig(bundle.path("figures/numerical_deviation.svg"))
    plt.close(fig)
    return "figures/numerical_deviation.svg"


def fig_symmetry(bundle, metrics) -> str | None:
    sym = metrics.get("symmetry") or {}
    neuron = sym.get("neuron_level") or []
    if not neuron:
        return None
    n = neuron[0]
    plt = _plt()
    so = np.array(n["scores_original"])
    so_mapped = np.array(n["scores_original_mapped"])  # so[perm]: aligned with transformed ids
    st = np.array(n["scores_transformed"])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    axes[0].scatter(so_mapped, st, s=8, color=BLUE)
    axes[0].set_title(f"through the component map (rho {n['mapped_spearman']:.3f})", fontsize=10, loc="left")
    axes[1].scatter(so, st, s=8, color=ORANGE)
    axes[1].set_title(f"naive ids (rho {n['naive_spearman']:.3f})", fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("original model, per-channel attribution")
    axes[0].set_ylabel("permuted model attribution")
    fig.suptitle(f"Layer {n['layer']} MLP channels after a function-preserving permutation",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(bundle.path("figures/symmetry_neurons.svg"))
    plt.close(fig)
    return "figures/symmetry_neurons.svg"


def fig_search(bundle, metrics) -> str | None:
    cx = metrics.get("counterexamples") or {}
    eff = cx.get("efficiency") or []
    if not eff:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(len(eff))
    m = np.array([e["cx_per_1k_mean"] for e in eff])
    lo = m - np.array([e["cx_per_1k_ci95"][0] for e in eff])
    hi = np.array([e["cx_per_1k_ci95"][1] for e in eff]) - m
    ax.bar(x, m, color=[BLUE, ORANGE, AQUA][:len(eff)], width=0.55)
    ax.errorbar(x, m, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2)
    ax.set_xticks(x, [e["strategy"] for e in eff])
    ax.set_ylabel("counterexamples / 1k forwards")
    ax.set_title("Search efficiency at equal compute (mean over seeds, 95% CI)",
                 fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(bundle.path("figures/search_efficiency.svg"))
    plt.close(fig)
    return "figures/search_efficiency.svg"


def _cell(c) -> str:
    return str(c).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _split_row(row: str) -> list[str]:
    inner = row.strip()[1:-1] if row.strip().endswith("|") else row.strip()[1:]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}"
    return str(x)


def build_markdown(claim, metrics, manifest, figures) -> str:
    v = metrics["verdict"]
    L = []
    L.append(f"# Evidence report: `{claim.claim_id}`\n")
    L.append(f"**Status: `{v['status']}`**\n")
    for r in v["reasons"]:
        L.append(f"- {r}")
    L.append("")
    L.append(f"> {claim.hypothesis}\n")
    L.append(f"Mechanism: {claim.mechanism}  \nUnit: {claim.unit}  \nModel: `{claim.model}` @ "
             f"`{claim.model_revision}`  \nScope: {claim.scope.input_distribution} "
             f"({claim.scope.dtype}, backends {', '.join(claim.scope.backends)})\n")

    cx = metrics.get("counterexamples") or {}
    L.append("## Smallest contradicting experiment\n")

    def prompt_block(d):
        L.append("```text")
        L.append(f"clean:    {d.get('clean')}")
        L.append(f"corrupt:  {d.get('corrupt')}")
        L.append(f"clean margin {d['clean_margin']:+.3f}   corrupt margin {d['corrupt_margin']:+.3f}   "
                 f"patched margin {d['patched_margin']:+.3f}   restoration {d['restoration']:.3f}")
        L.append("```\n")

    base = cx.get("base_input_evaluation") or {}
    if base.get("violates"):
        L.append(f"Prediction `{cx['prediction']}` requires patching MLP blocks {cx['components']} to restore "
                 f"at least {cx['threshold']} of the contrast on every valid input. The base input, the "
                 f"first point of the search space with nothing varied, is already valid and already "
                 f"fails it, so no shrinking is needed:\n")
        prompt_block(base["detail"])
        L.append(f"Search found {cx['unique_counterexamples']} distinct counterexamples, "
                 f"{cx['reproduced']} of which reproduced on a fresh evaluation (`counterexamples.jsonl`).\n")
    elif cx.get("minimized"):
        best = min(cx["minimized"], key=lambda m: (len(m["changed_axes"]), m["evaluation"]["score"]))
        L.append(f"Prediction `{cx['prediction']}` requires patching MLP blocks {cx['components']} to restore "
                 f"at least {cx['threshold']} of the contrast on every valid input. This input is valid "
                 f"(the model does the task) and fails it. It differs from the base input only in "
                 f"`{best['changed_axes']}`.\n")
        prompt_block(best["evaluation"]["detail"])
        for s in cx.get("slices", []):
            L.append(f"- Holding `{s['fixed']}` fixed and randomising everything else: "
                     f"{s['violations']}/{s['valid']} valid inputs fail "
                     f"(Wilson 95% [{s['wilson95'][0]:.2f}, {s['wilson95'][1]:.2f}]; "
                     f"claim allows {cx['max_violation_rate']}).")
        L.append("")
        L.append(f"Replay: every counterexample is stored in `counterexamples.jsonl`; "
                 f"{cx['reproduced']}/{cx['unique_counterexamples']} reproduced on a fresh evaluation.\n")
    elif cx:
        L.append(f"Search found {cx.get('unique_counterexamples', 0)} counterexamples. "
                 + (cx.get("note") or "None were minimized.") + "\n")
    else:
        L.append("Counterexample search was not run for this claim.\n")

    L.append("## Predictions\n")
    L.append(_table(["id", "kind", "split", "outcome", "evidence"],
                    [[p["id"], p["kind"], p["split"], f"**{p['outcome']}**", p["reason"]] for p in v["predictions"]]))
    L.append("")

    L.append("## Gates\n")
    L.append(_table(["gate", "ok", "detail"], [[k, g["ok"], g["detail"]] for k, g in metrics["gates"].items()]))
    L.append("")
    L.append("## Denominator\n")
    rows = []
    for split, d in metrics["denominator"].items():
        rej = ", ".join(f"{k}: {n}" for k, n in sorted(d["rejections"].items())) or "none"
        rows.append([split, d["generated"], d["valid"], rej])
    L.append(_table(["split", "generated", "valid", "rejections (a case can have several)"], rows))
    L.append("")

    L.append("## Baseline: single-block MLP patching\n")
    if figures.get("layers"):
        L.append(f"![layer restoration]({figures['layers']})\n")
    rows = [[r["split"], r["layer"], _f(r["restoration"]["mean"]),
             f"[{_f(r['restoration']['ci95_low'])}, {_f(r['restoration']['ci95_high'])}]",
             _f(r["effect"]["mean"]), r["restoration"]["n"], r["restoration"]["n_clusters"]]
            for r in metrics.get("layers", [])]
    L.append(_table(["split", "layer", "mean restoration", "95% CI", "mean effect (logits)", "n", "templates"], rows))
    L.append("")
    c = metrics.get("controls") or {}
    if c:
        s = c["heldout_fidelity_selected"]
        L.append("### Controls\n")
        L.append(f"- Identity patch (corrupt activation into corrupt run): max deviation "
                 f"{_f(metrics.get('identity_patch_max_deviation'))} logits.")
        L.append(f"- Top-{len(c['discovery_selected_top_k'])} blocks selected on discovery: "
                 f"{c['discovery_selected_top_k']}. Held-out joint restoration {_f(s['mean'])} "
                 f"[{_f(s['ci95_low'])}, {_f(s['ci95_high'])}], above "
                 f"{100 * (c['heldout_fidelity_percentile_vs_random'] or 0):.0f}% of "
                 f"{len(c['heldout_fidelity_random_sets'])} random same-size block sets.")
        for row in c["claimed_vs_random"]:
            L.append(f"- Claimed blocks {row['components']} on {row['split']}: mean restoration "
                     f"{_f(row['mean_restoration'])} vs random same-size sets mean {_f(row['random_mean'])}, "
                     f"max {_f(row['random_max'])} (n={row['n_random']}).")
        L.append("")

    num = metrics.get("numerical")
    if num:
        L.append("## Challenge: numerical execution\n")
        L.append("Every condition changes one factor relative to the reference. Measurements use the "
                 "reference's set of valid cases so the denominator never moves.\n")
        if figures.get("numerical"):
            L.append(f"![numerical deviation]({figures['numerical']})\n")
        rows = []
        for r in num:
            if not r.get("valid"):
                rows.append([r["condition"], "+".join(r["factors"]), r.get("observed_attention", "n/a"),
                             "**not run**", r.get("problem", ""), "", "", ""])
                continue
            par = r.get("output_parity") or {}
            rows.append([r["condition"], "+".join(r["factors"]), r.get("observed_attention"),
                         r["classification"], _f(r["max_abs_patched_margin_diff"]),
                         _f(par.get("max_abs_logit_diff")),
                         f"{r['discovery_top_k_condition']}{' (near tie)' if r['selection_near_tie'] else ''}",
                         (r["prediction_flips"] or "none")
                         if not r.get("cases_with_undefined_restoration")
                         else f"{r['prediction_flips'] or 'none'}; {r['cases_with_undefined_restoration']} "
                              f"cases with zero contrast"])
        L.append(_table(["condition", "factor", "observed attention", "classification",
                         "max d patched margin", "max d clean logit", "discovery top-k", "prediction flips"], rows))
        L.append("")

    sym = metrics.get("symmetry")
    if sym:
        L.append("## Challenge: function-preserving MLP channel permutation\n")
        par = sym.get("parity", {})
        L.append(f"Every MLP's hidden channels permuted (seed {sym['seed']}). Output parity over "
                 f"{par.get('scope')}: max |d logit| {_f(par.get('max_abs_logit_diff'))}, argmax agreement "
                 f"{_f(par.get('argmax_agreement'))}, gate {'passed' if par.get('passed') else 'FAILED'} "
                 f"(tolerance atol + rtol * |logit| with rtol {sym.get('parity_rtol', 0)}) "
                 f"at atol {sym['parity_atol']}.\n")
        if sym.get("comparisons"):
            L.append(f"**Comparisons {sym['comparisons']}.** The challenge counts as not validly run, so "
                     f"the mlp_channel_permutation invariance stays untested.
")
        if "block_level" in sym:
            b = sym["block_level"]
            L.append(f"Block-level effects on the permuted model: {b['classification']}, max |d patched margin| "
                     f"{_f(b['max_abs_patched_margin_diff'])}, prediction flips {b['prediction_flips'] or 'none'}.\n")
        if figures.get("symmetry"):
            L.append(f"![symmetry]({figures['symmetry']})\n")
        rows = [[n["layer"], _f(n["mapped_spearman"], 4), _f(n["naive_spearman"], 4),
                 _f(n["exact_patch_restoration_original"], 4), _f(n["exact_patch_restoration_mapped"], 4),
                 _f(n["exact_patch_restoration_naive_ids"], 4), n["naive_labels_flagged"]]
                for n in sym.get("neuron_level", [])]
        if rows:
            L.append(_table(["layer", "rho mapped", "rho naive ids", "top-k patch: original",
                             "permuted, mapped ids", "permuted, naive ids", "naive ids flagged"], rows))
            L.append("")
            L.append("A neuron-level claim survives the permutation only through the component map. "
                     "Read with naive ids, the same claim points at unrelated channels.\n")

    if cx:
        L.append("## Challenge: counterexample search\n")
        if figures.get("search"):
            L.append(f"![search]({figures['search']})\n")
        L.append(_table(["strategy", "seeds", "counterexamples / 1k forwards", "95% CI", "valid proposal rate",
                         "lowest restoration reached (mean over seeds)"],
                        [[e["strategy"], e["seeds"], _f(e["cx_per_1k_mean"]),
                          f"[{_f(e['cx_per_1k_ci95'][0])}, {_f(e['cx_per_1k_ci95'][1])}]",
                          _f(e["valid_proposal_rate"]), _f(e.get("lowest_restoration_mean"))]
                         for e in cx["efficiency"]]))
        L.append("")
        L.append(f"The claim's per-case threshold is {cx['threshold']}. The last column shows how close each "
                 f"strategy pushed the claim towards failing within its budget of "
                 f"{cx['runs'][0]['budget_forwards']} forwards per seed, which matters most when no "
                 f"counterexample was found.\n")
        L.append("Counterexamples are drawn from the discovery axis space only. The holdout templates and "
                 "names are never searched.\n")

    L.append("## Challenge findings\n")
    L.append(_table(["challenge", "invariance", "valid", "contradicts", "summary"],
                    [[f["name"], f["invariance"], f["valid"], f["contradicts"], f["summary"]] for f in v["findings"]]))
    L.append("")
    L.append("## Limitations\n")
    L.append("- Block-level MLP patching over all positions. It localises where restoring a clean activation "
             "matters; it does not identify the algorithm, and attention heads are not tested here.")
    L.append("- One task family (IOI with an S2-flip corruption). A status holds only for the scope above.")
    L.append("- Template-cluster bootstrap with few templates gives coarse intervals. The cluster count is in "
             "the tables.")
    L.append("- Kernel dispatch is read from the profiler for one batch per condition and assumed stable "
             "across batches of the same shape.")
    L.append("")
    env = manifest["environment"]
    L.append("## Reproduce\n")
    L.append("```bash")
    L.append(manifest["reproduce"])
    L.append("```\n")
    L.append(f"torch {env['torch']} (CUDA {env['torch_cuda']}), transformers {env['transformers']}, "
             f"{env.get('gpu_name', 'CPU')}, driver {env.get('nvidia_driver')}, git {str(env['git_commit'])[:10]}"
             f"{' (dirty)' if env['git_dirty'] else ''}.\n")
    return "\n".join(L)


def markdown_to_html(md: str, title: str) -> str:
    """Small converter for the subset build_markdown emits."""
    out, in_code, in_list, table = [], False, False, []

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return s

    def flush_table():
        if not table:
            return
        head = _split_row(table[0])
        out.append("<table><thead><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr></thead><tbody>")
        for row in table[2:]:
            cells = _split_row(row)
            out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
        out.append("</tbody></table>")
        table.clear()

    for line in md.splitlines():
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.strip():
            out.append(f"<p>{inline(line.rstrip()).replace('  ', '<br>')}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--code:#f2f1ee}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--code:#262624}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px}
h1{font-size:1.5rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}code,pre{background:var(--code);border-radius:4px}
code{padding:1px 4px}pre{padding:12px;overflow-x:auto;font-size:13px}
img{max-width:100%;background:#fcfcfb;border-radius:6px}blockquote{color:var(--ink2);border-left:3px solid var(--line);margin:0;padding-left:12px}
"""
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body>{''.join(out)}</body></html>")


def write_report(bundle, claim, metrics, manifest) -> None:
    figures = {
        "layers": fig_layers(bundle, metrics),
        "numerical": fig_numerical(bundle, metrics),
        "symmetry": fig_symmetry(bundle, metrics),
        "search": fig_search(bundle, metrics),
    }
    md = build_markdown(claim, metrics, manifest, figures)
    bundle.text("report.md", md)
    bundle.text("report.html", markdown_to_html(md, f"KernelTrace: {claim.claim_id}"))


def rebuild(bundle_dir) -> None:
    """Regenerate report and figures from a bundle's saved metrics, without rerunning anything."""
    import json
    from pathlib import Path

    from ..core.spec import load_claim
    from .bundle import Bundle

    root = Path(bundle_dir)
    bundle = Bundle(root)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    write_report(bundle, load_claim(root / "claim.yaml"), metrics, manifest)
    bundle.json("files.json", bundle.file_hashes())
