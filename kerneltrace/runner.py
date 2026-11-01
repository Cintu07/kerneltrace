"""End-to-end experiment: claim card in, evidence bundle out."""
from __future__ import annotations

import gc
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import verdict as V
from .core.manifest import (configure_determinism, environment, new_run_id, numerical_state,
                            resolve_device, sha256_file, sha256_json)
from .core.spec import ConditionSpec, NumericalSpec, Prediction, dump_yaml, load_experiment
from .interventions.patching import (Batch, block_effects, joint_patch_margins, make_batches, margins,
                                     neuron_attribution, neuron_patch_margins, run)
from .metrics import stats as S
from .models.adapter import ModelAdapter, load, load_tokenizer
from .reports.bundle import Bundle
from .search import counterexample as CX
from .tasks import ioi
from .tasks.preflight import TokenizedCase, behavioural_check, single_token, structural_check
from .transforms import backend as B
from .transforms import symmetry as SYM

IDENTITY_TOL = 1e-6
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def log(msg: str) -> None:
    print(f"[kerneltrace {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _invariance_for(factors: list[str]) -> str:
    """Map a single numerical factor to the claim invariance it tests.

    A dtype change is a different measurement, not an invariance of an fp32
    claim, so it is reported but cannot challenge the claim.
    """
    if len(factors) != 1:
        return "out_of_scope"
    return factors[0] if factors[0] in ("attention_backend", "batching", "repeat", "tf32") else "out_of_scope"


def _neuron_restoration(adapter: ModelAdapter, batches: list[Batch], layer: int,
                        channels: torch.Tensor) -> np.ndarray:
    """Per-case restoration from patching a set of hidden channels, using this model's own margins."""
    out = []
    for b in batches:
        clean = margins(run(adapter, b.clean)[0], b.a_ids, b.b_ids).cpu().numpy()
        corrupt = margins(run(adapter, b.corrupt)[0], b.a_ids, b.b_ids).cpu().numpy()
        patched = neuron_patch_margins(adapter, b, layer, channels)
        out.append((patched - corrupt) / (clean - corrupt))
    return np.concatenate(out)


@dataclass
class Measurement:
    """Every per-case number one condition produced, in a fixed case order."""

    condition: str
    case_ids: list[str]
    clean: np.ndarray                 # [N]
    corrupt: np.ndarray               # [N]
    patched: np.ndarray               # [N, L]
    identity: np.ndarray | None       # [N, L]
    joint: dict = field(default_factory=dict)  # components tuple -> patched margin [N]
    clean_last_logits: torch.Tensor | None = None  # [N, V]

    @property
    def effects(self) -> np.ndarray:
        return self.patched - self.corrupt[:, None]

    def _contrast(self) -> np.ndarray:
        """Clean minus corrupt margin, with exact zeros as NaN so restoration is undefined, not inf."""
        c = self.clean - self.corrupt
        return np.where(c == 0, np.nan, c)

    def restoration(self) -> np.ndarray:
        return (self.patched - self.corrupt[:, None]) / self._contrast()[:, None]

    def joint_restoration(self, comps: tuple) -> np.ndarray:
        return (self.joint[comps] - self.corrupt) / self._contrast()


def _order(cases: list[TokenizedCase], batches: list[Batch]) -> list[int]:
    pos = {c.case.case_id: i for i, c in enumerate(cases)}
    return [pos[cid] for b in batches for cid in b.case_ids]


def measure(adapter: ModelAdapter, cases: list[TokenizedCase], batch_size: int,
            condition: ConditionSpec, component_sets: list[tuple], identity_controls: bool,
            keep_logits: bool = True) -> Measurement:
    batches = make_batches(cases, batch_size, adapter.device)
    order = _order(cases, batches)
    clean, corrupt, patched, identity, logits = [], [], [], [], []
    joint = {c: [] for c in component_sets}
    with B.execution(condition):
        for batch in batches:
            be = block_effects(adapter, batch, identity_controls)
            clean.append(be.clean)
            corrupt.append(be.corrupt)
            patched.append(be.patched.T)
            if be.identity is not None:
                identity.append(be.identity.T)
            if keep_logits:
                logits.append(be.clean_last_logits)
            for comps in component_sets:
                joint[comps].append(joint_patch_margins(adapter, batch, list(comps)))
    inv = np.argsort(order)

    def cat(parts):
        return np.concatenate(parts)[inv]

    return Measurement(
        condition=condition.name,
        case_ids=[c.case.case_id for c in cases],
        clean=cat(clean), corrupt=cat(corrupt), patched=cat(patched),
        identity=cat(identity) if identity else None,
        joint={c: cat(v) for c, v in joint.items()},
        clean_last_logits=torch.cat(logits)[torch.as_tensor(inv)] if keep_logits else None,
    )


def add_joint(meas: Measurement, adapter: ModelAdapter, cases: list[TokenizedCase], batch_size: int,
              condition: ConditionSpec, component_sets: list[tuple]) -> None:
    todo = [c for c in component_sets if c not in meas.joint]
    if not todo:
        return
    batches = make_batches(cases, batch_size, adapter.device)
    inv = np.argsort(_order(cases, batches))
    with B.execution(condition):
        for comps in todo:
            parts = [joint_patch_margins(adapter, b, list(comps)) for b in batches]
            meas.joint[comps] = np.concatenate(parts)[inv]


class Experiment:
    def __init__(self, spec_path: str | Path, out_root: str | Path = "runs"):
        self.spec_path = Path(spec_path)
        self.spec, self.claim = load_experiment(self.spec_path)
        self.spec_hash = sha256_json(self.spec.model_dump(mode="json") | {"claim": self.claim.model_dump(mode="json")})
        self.run_id = new_run_id(self.spec.experiment_id, self.spec_hash)
        self.bundle = Bundle(Path(out_root) / self.run_id)
        self.findings: list[V.ChallengeFinding] = []
        self.gates: dict[str, tuple[bool, str]] = {}
        self.results: dict = {}
        self.raw_frames: list[pd.DataFrame] = []
        self.interventions: list[dict] = []

    # ---------------------------------------------------------------- setup
    @property
    def ref_condition(self) -> ConditionSpec:
        num = self.spec.challenges.numerical
        if num is not None and num.enabled:
            ref = num.reference
            if (ref.attn_implementation, ref.dtype) != (self.spec.attn_implementation, self.spec.dtype):
                raise ValueError("numerical.reference must match the experiment's attn_implementation and dtype")
            return ref
        return ConditionSpec(name=f"{self.spec.attn_implementation}_{self.spec.dtype}",
                             attn_implementation=self.spec.attn_implementation, dtype=self.spec.dtype)

    def claim_sets(self) -> list[tuple]:
        sets = []
        for p in self.claim.predictions:
            c = tuple(p.components)
            if p.kind in ("restoration_at_least", "effect_positive") and c not in sets:
                sets.append(c)
        return sets

    def setup(self) -> None:
        spec = self.spec
        self.determinism = configure_determinism(spec.seed, spec.deterministic)
        self.device = resolve_device(spec.device)
        log(f"run {self.run_id} on {self.device}")
        self.tokenizer = load_tokenizer(spec.model, spec.model_revision)
        self.adapter = load(spec.model, spec.model_revision, spec.attn_implementation, spec.dtype, self.device)
        for p in self.claim.predictions:
            bad = [c for c in p.components if c >= self.adapter.n_layers]
            if bad:
                raise ValueError(f"prediction {p.id} names layers {bad} but the model has {self.adapter.n_layers}")
        commit = self.adapter.describe()["commit"]
        pinned = bool(HEX40.match(spec.model_revision)) and commit == spec.model_revision
        self.gates["model_revision_pinned"] = (
            pinned, f"revision {spec.model_revision!r}, loaded commit {commit!r}")

    # ---------------------------------------------------------------- tasks
    def names_ok(self, name: str) -> bool:
        return single_token(self.tokenizer, " " + name) is not None

    def build_tasks(self) -> None:
        t = self.spec.task
        generated = (ioi.generate("discovery", t.n_discovery, self.spec.seed, self.names_ok)
                     + ioi.generate("holdout", t.n_holdout, self.spec.seed, self.names_ok))
        self.preflight_rows, self.cases = [], []
        for case in generated:
            tok, reasons = structural_check(case, self.tokenizer, t.max_seq_len)
            self.preflight_rows.append(dict(
                case_id=case.case_id, split=case.split, template=case.template, axes=case.axes,
                clean=case.clean, corrupt=case.corrupt, answer=case.answer, distractor=case.distractor,
                structural_reasons=reasons, behavioural_reasons=None, valid=False))
            if tok is not None:
                self.cases.append(tok)
        excluded = [n for n in ioi.NAMES if not self.names_ok(n)]
        self.results["names_excluded_not_single_token"] = excluded
        log(f"{len(generated)} cases generated, {len(self.cases)} pass structural checks")

    def apply_behaviour(self, meas: Measurement) -> np.ndarray:
        t = self.spec.task
        rows = {r["case_id"]: r for r in self.preflight_rows}
        valid = np.zeros(len(self.cases), bool)
        for i, c in enumerate(self.cases):
            reasons = behavioural_check(float(meas.clean[i]), float(meas.corrupt[i]),
                                        t.min_contrast, t.require_corrupt_flip)
            r = rows[c.case.case_id]
            r.update(behavioural_reasons=reasons, valid=not reasons,
                     clean_margin=float(meas.clean[i]), corrupt_margin=float(meas.corrupt[i]))
            valid[i] = not reasons
        return valid

    def split_mask(self, split: str) -> np.ndarray:
        return np.array([c.case.split == split for c in self.cases])

    def split_data(self, meas: Measurement, valid: np.ndarray, split: str) -> V.SplitData:
        m = valid & self.split_mask(split)
        clusters = np.array([c.case.template for c in self.cases])[m]
        return V.SplitData(
            split=split, clusters=clusters, effects=meas.effects[m],
            joint_restoration={k: meas.joint_restoration(k)[m] for k in meas.joint},
            joint_effect={k: (meas.joint[k] - meas.corrupt)[m] for k in meas.joint},
        )

    def evaluate_predictions(self, meas: Measurement, valid: np.ndarray) -> list[V.PredictionResult]:
        b = self.spec.baseline
        data = {s: self.split_data(meas, valid, s) for s in ("discovery", "holdout")}
        return [V.evaluate_prediction(p, data[p.split], b.bootstrap, self.spec.seed, self.spec.task.min_valid)
                for p in self.claim.predictions]

    def raw_frame(self, meas: Measurement, valid: np.ndarray) -> pd.DataFrame:
        n, L = meas.patched.shape
        cases = self.cases
        contrast = meas.clean - meas.corrupt
        safe = np.where(np.abs(contrast) < 1e-9, np.nan, contrast)
        df = pd.DataFrame({
            "condition": meas.condition,
            "case_id": np.repeat(meas.case_ids, L),
            "split": np.repeat([c.case.split for c in cases], L),
            "template": np.repeat([c.case.template for c in cases], L),
            "valid": np.repeat(valid, L),
            "layer": np.tile(np.arange(L), n),
            "clean_margin": np.repeat(meas.clean, L),
            "corrupt_margin": np.repeat(meas.corrupt, L),
            "patched_margin": meas.patched.reshape(-1),
            "effect": meas.effects.reshape(-1),
            "restoration": (meas.effects / safe[:, None]).reshape(-1),
        })
        if meas.identity is not None:
            df["identity_patched_margin"] = meas.identity.reshape(-1)
        return df

    # ---------------------------------------------------------------- baseline
    def baseline(self) -> None:
        spec, ref = self.spec, self.ref_condition
        t, b = spec.task, spec.baseline
        log("baseline: probing dispatch and measuring all MLP blocks")
        first = make_batches(self.cases, t.batch_size, self.adapter.device)[0]
        self.ref_probe = B.probe(self.adapter, first.clean, ref)
        self.bundle.json(f"profiler/{ref.name}.json", self.ref_probe)
        self.gates["reference_dispatch_verified"] = (self.ref_probe["verified"], "; ".join(self.ref_probe["problems"]) or "ok")

        meas = measure(self.adapter, self.cases, t.batch_size, ref, self.claim_sets(), b.identity_controls)
        self.ref = meas
        self.valid = self.apply_behaviour(meas)
        n_valid = {s: int((self.valid & self.split_mask(s)).sum()) for s in ("discovery", "holdout")}
        ok = all(v >= t.min_valid for v in n_valid.values())
        self.gates["task_validity"] = (ok, f"valid cases per split {n_valid}, need {t.min_valid}")
        log(f"valid cases {n_valid}")
        if b.identity_controls:
            dev = float(np.abs(meas.identity - meas.corrupt[:, None]).max())
            self.gates["identity_patch_null"] = (dev <= IDENTITY_TOL, f"max |identity patch - corrupt| = {dev:.3g}")
            self.results["identity_patch_max_deviation"] = dev
        self.interventions.append(dict(condition=ref.name, kind="single_block_mlp_out_patch",
                                       sites=[f"mlp_out.{l}" for l in range(self.adapter.n_layers)],
                                       positions="all", source="clean", target="corrupt",
                                       n_cases=len(self.cases)))
        for comps in self.claim_sets():
            self.interventions.append(dict(condition=ref.name, kind="joint_mlp_out_patch",
                                           sites=[f"mlp_out.{l}" for l in comps], positions="all",
                                           source="clean", target="corrupt", n_cases=len(self.cases)))

        # per-layer summaries
        layer_rows = []
        for split in ("discovery", "holdout"):
            m = self.valid & self.split_mask(split)
            if m.sum() < 2:
                continue
            clusters = np.array([c.case.template for c in self.cases])[m]
            rest = meas.restoration()[m]
            for layer in range(self.adapter.n_layers):
                s = S.summarize(rest[:, layer], clusters, b.bootstrap, spec.seed + layer)
                e = S.summarize(meas.effects[m, layer], clusters, b.bootstrap, spec.seed + layer)
                layer_rows.append(dict(split=split, layer=layer, restoration=s, effect=e))
        self.results["layers"] = layer_rows

        self.controls()
        self.predictions = self.evaluate_predictions(meas, self.valid)
        self.raw_frames.append(self.raw_frame(meas, self.valid))
        for p in self.predictions:
            log(f"prediction {p.id}: {p.outcome} ({p.reason})")

    def controls(self) -> None:
        """Random matched-size circuits and discovery-selected held-out fidelity."""
        spec, b, ref = self.spec, self.spec.baseline, self.ref_condition
        L = self.adapter.n_layers
        rng = random.Random(f"controls:{spec.seed}")
        disc = self.valid & self.split_mask("discovery")
        hold = self.valid & self.split_mask("holdout")
        if disc.sum() < 2 or hold.sum() < 2:
            return
        selected = tuple(S.top_k(self.ref.effects[disc].mean(0), b.fidelity_top_k))
        sizes = sorted({len(c) for c in self.claim_sets()} | {len(selected)})
        random_sets = {}
        for size in sizes:
            pool = set()
            tries = 0
            while len(pool) < b.random_circuit_controls and tries < 50 * b.random_circuit_controls:
                tries += 1
                pool.add(tuple(sorted(rng.sample(range(L), size))))
            random_sets[size] = sorted(pool)
        all_sets = [selected] + [s for v in random_sets.values() for s in v]
        add_joint(self.ref, self.adapter, self.cases, spec.task.batch_size, ref, all_sets)

        clusters = np.array([c.case.template for c in self.cases])

        def mean_rest(comps, mask):
            return float(self.ref.joint_restoration(comps)[mask].mean())

        out = {"discovery_selected_top_k": list(selected)}
        s = S.summarize(self.ref.joint_restoration(selected)[hold], clusters[hold], b.bootstrap, spec.seed)
        null = [mean_rest(c, hold) for c in random_sets[len(selected)] if c != selected]
        out["heldout_fidelity_selected"] = s
        out["heldout_fidelity_random_sets"] = null
        out["heldout_fidelity_percentile_vs_random"] = float(np.mean([s["mean"] > x for x in null])) if null else None
        claimed = []
        for comps in self.claim_sets():
            for split, mask in (("discovery", disc), ("holdout", hold)):
                val = mean_rest(comps, mask)
                null = [mean_rest(c, mask) for c in random_sets[len(comps)] if c != comps]
                claimed.append(dict(components=list(comps), split=split, mean_restoration=val,
                                    random_mean=float(np.mean(null)) if null else None,
                                    random_max=float(np.max(null)) if null else None,
                                    percentile_vs_random=float(np.mean([val > x for x in null])) if null else None,
                                    n_random=len(null)))
        out["claimed_vs_random"] = claimed
        self.results["controls"] = out
        log(f"discovery-selected top-{b.fidelity_top_k} layers {list(selected)}; held-out restoration {s['mean']:.3f} "
            f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]")

    # ---------------------------------------------------------------- numerical
    def compare(self, alt: Measurement, alt_preds: list[V.PredictionResult], cfg) -> dict:
        ref, valid = self.ref, self.valid
        m = valid
        d_patch = np.abs(alt.patched[m] - ref.patched[m])
        tol = cfg.effect_atol + cfg.effect_rtol * np.abs(ref.patched[m])
        d_eff = np.abs(alt.effects[m] - ref.effects[m])
        d_rest = np.abs(alt.restoration()[m] - ref.restoration()[m])
        undefined = int((~np.isfinite(alt.restoration()[m]).all(axis=1)).sum())
        d_rest = d_rest[np.isfinite(d_rest)] if np.isfinite(d_rest).any() else np.array([np.nan])
        bitwise = (np.array_equal(alt.clean, ref.clean) and np.array_equal(alt.corrupt, ref.corrupt)
                   and np.array_equal(alt.patched, ref.patched))
        within = bool((d_patch <= tol).all()) and bool(
            (np.abs(alt.clean[m] - ref.clean[m]) <= cfg.effect_atol + cfg.effect_rtol * np.abs(ref.clean[m])).all())
        disc = valid & self.split_mask("discovery")
        k = self.spec.baseline.fidelity_top_k
        sel_ref = S.top_k(ref.effects[disc].mean(0), k)
        sel_alt = S.top_k(alt.effects[disc].mean(0), k)
        ties = S.topk_with_ties(ref.effects[disc].mean(0), alt.effects[disc].mean(0), k, tie_tol=1e-3)
        per_case_rho = [S.spearman(a, r) for a, r in zip(alt.effects[m], ref.effects[m])]
        per_case_rho = [x for x in per_case_rho if x is not None]
        per_case_top3 = [S.topk_overlap(list(a), list(r), 3) for a, r in zip(alt.effects[m], ref.effects[m])]
        ref_out = {p.id: p.outcome for p in self.predictions}
        alt_out = {p.id: p.outcome for p in alt_preds}
        flips = [pid for pid in ref_out if ref_out[pid] != alt_out[pid]]
        t = self.spec.task
        validity_flips = sum(
            bool(behavioural_check(float(alt.clean[i]), float(alt.corrupt[i]), t.min_contrast, t.require_corrupt_flip))
            != bool(not valid[i]) for i in range(len(valid)))
        parity = SYM.logit_parity(ref.clean_last_logits, alt.clean_last_logits, atol=cfg.effect_atol) \
            if ref.clean_last_logits is not None and alt.clean_last_logits is not None else None
        selection_changed = sel_ref != sel_alt
        if bitwise:
            klass = "bitwise_identical"
        elif within:
            klass = "within_tolerance"
        elif flips or undefined or (selection_changed and not ties["near_tie"]):
            klass = "decision_changing"
        else:
            klass = "exceeds_tolerance_no_decision_change"
        return dict(
            classification=klass,
            bitwise_identical=bitwise,
            within_tolerance=within,
            max_abs_patched_margin_diff=float(d_patch.max()),
            mean_abs_patched_margin_diff=float(d_patch.mean()),
            max_abs_effect_diff=float(d_eff.max()),
            max_abs_restoration_diff=float(d_rest.max()),
            cases_with_undefined_restoration=undefined,
            frac_entries_outside_tolerance=float((d_patch > tol).mean()),
            output_parity=asdict(parity) | {"passed_at_effect_atol": parity.passed} if parity else None,
            per_case_effect_spearman_min=float(min(per_case_rho)) if per_case_rho else None,
            per_case_effect_spearman_mean=float(np.mean(per_case_rho)) if per_case_rho else None,
            per_case_top3_jaccard_mean=float(np.mean(per_case_top3)),
            per_case_top3_jaccard_min=float(np.min(per_case_top3)),
            discovery_top_k_reference=sel_ref,
            discovery_top_k_condition=sel_alt,
            selection_changed=selection_changed,
            selection_near_tie=ties["near_tie"],
            prediction_outcomes=alt_out,
            prediction_flips=flips,
            behavioural_validity_flips=int(validity_flips),
        )

    def numerical(self) -> None:
        cfg = self.spec.challenges.numerical
        if cfg is None or not cfg.enabled:
            return
        ref = self.ref_condition
        t = self.spec.task
        rows = []
        for cond in cfg.conditions:
            fac = B.factors(ref, cond)
            log(f"numerical: condition {cond.name} (factor {'+'.join(fac)})")
            row = dict(condition=cond.name, factors=fac, requested=cond.model_dump())
            if len(fac) > 1 and not cfg.allow_multi_factor:
                row.update(valid=False, problem=f"changes {fac} at once; FR-04 forbids mixed factors")
                rows.append(row)
                continue
            same_model = (cond.attn_implementation, cond.dtype) == (ref.attn_implementation, ref.dtype)
            adapter = self.adapter if same_model else load(
                self.spec.model, self.spec.model_revision, cond.attn_implementation, cond.dtype, self.device)
            try:
                first = make_batches(self.cases, t.batch_size, adapter.device)[0]
                try:
                    prb = B.probe(adapter, first.clean, cond)
                except RuntimeError as exc:  # e.g. a forced kernel that cannot run this dtype
                    prb = dict(condition=cond.name, verified=False, problems=[f"probe raised: {exc}"],
                               observed_attention="none")
                self.bundle.json(f"profiler/{cond.name}.json", prb)
                row["observed_attention"] = prb.get("observed_attention")
                if not prb["verified"]:
                    row.update(valid=False, problem="; ".join(prb["problems"]))
                    rows.append(row)
                    log(f"  invalid: {row['problem']}")
                    continue
                meas = measure(adapter, self.cases, cond.batch_size or t.batch_size, cond,
                               self.claim_sets(), identity_controls=False)
                meas.condition = cond.name
                add_joint(meas, adapter, self.cases, cond.batch_size or t.batch_size, cond,
                          [tuple(self.results["controls"]["discovery_selected_top_k"])] if "controls" in self.results else [])
                preds = self.evaluate_predictions(meas, self.valid)
                cmp = self.compare(meas, preds, cfg)
                row.update(valid=True, **cmp)
                self.raw_frames.append(self.raw_frame(meas, self.valid))
                self.interventions.append(dict(condition=cond.name, kind="single_block_mlp_out_patch",
                                               sites="all mlp_out", n_cases=len(self.cases)))
                log(f"  {cmp['classification']}: max |d patched margin| {cmp['max_abs_patched_margin_diff']:.3g}, "
                    f"flips {cmp['prediction_flips']}, selection {cmp['discovery_top_k_condition']}")
            finally:
                if adapter is not self.adapter:
                    del adapter
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            rows.append(row)
            inv = _invariance_for(fac)
            contradicts = row["classification"] == "decision_changing"
            self.findings.append(V.ChallengeFinding(
                invariance=inv, name=f"numerical:{cond.name}", valid=True, contradicts=contradicts,
                summary=(f"{row['classification']}; prediction flips {row['prediction_flips']}; "
                         f"top-k {row['discovery_top_k_reference']} -> {row['discovery_top_k_condition']}"
                         f"{' (near tie)' if row['selection_near_tie'] else ''}")))
        for row in rows:
            if not row.get("valid"):
                inv = _invariance_for(row["factors"])
                self.findings.append(V.ChallengeFinding(inv, f"numerical:{row['condition']}", False, False,
                                                        f"not run: {row.get('problem')}"))
        self.results["numerical"] = rows

    # ---------------------------------------------------------------- symmetry
    def symmetry(self) -> None:
        cfg = self.spec.challenges.symmetry
        if cfg is None or not cfg.enabled:
            return
        ref, t = self.ref_condition, self.spec.task
        log("symmetry: permuting every MLP's hidden channels")
        perms = SYM.random_permutations(self.adapter.n_layers, self.adapter.d_mlp, cfg.seed)
        pm = SYM.permuted_copy(self.adapter, perms)
        out: dict = {"seed": cfg.seed, "layers_permuted": list(perms), "parity_atol": cfg.parity_atol,
                     "parity_rtol": cfg.parity_rtol}
        self.results["symmetry"] = out  # filled in below; kept even if the parity gate stops the challenge
        try:
            idx = np.flatnonzero(self.valid)[:cfg.n_cases]
            sub = [self.cases[i] for i in idx]
            batches = make_batches(sub, t.batch_size, self.adapter.device)
            worst = None
            with B.execution(ref):
                for b in batches:
                    for ids in (b.clean, b.corrupt):
                        la, _ = run(self.adapter, ids)
                        lb, _ = run(pm.adapter, ids)
                        p = SYM.logit_parity(la, lb, cfg.parity_atol, cfg.parity_rtol)
                        if worst is None or p.max_excess_over_tolerance > worst.max_excess_over_tolerance:
                            worst = p
            out["parity"] = asdict(worst) | {"passed": worst.passed, "n_cases": len(sub),
                                             "scope": "all positions, full vocabulary, clean and corrupt"}
            if not worst.passed:
                out["comparisons"] = "skipped: output parity failed, the transform is not validated"
                self.findings.append(V.ChallengeFinding("mlp_channel_permutation", "symmetry", False, False,
                                                        f"parity failed: max |dlogit| {worst.max_abs_logit_diff:.3g}"))
                log(f"  parity FAILED: max |dlogit| {worst.max_abs_logit_diff:.3g} exceeds atol {cfg.parity_atol} "
                    f"+ rtol {cfg.parity_rtol} * |logit|; "
                    f"permuted model not compared")
                return
            log(f"  parity ok: max |dlogit| {worst.max_abs_logit_diff:.3g}")

            meas = measure(pm.adapter, self.cases, t.batch_size, ref, self.claim_sets(), identity_controls=False)
            meas.condition = "mlp_permuted"
            preds = self.evaluate_predictions(meas, self.valid)
            cmp = self.compare(meas, preds, self.spec.challenges.numerical or NumericalSpec())
            out["block_level"] = cmp
            self.raw_frames.append(self.raw_frame(meas, self.valid))

            neuron = []
            k = cfg.top_k_neurons
            for layer in cfg.neuron_layers:
                with B.execution(ref):
                    so = np.concatenate([neuron_attribution(self.adapter, b, layer) for b in batches]).mean(0)
                    st = np.concatenate([neuron_attribution(pm.adapter, b, layer) for b in batches]).mean(0)
                    cmpn = SYM.mapped_comparison(so, st, perms[layer])
                    top_orig = torch.as_tensor(np.argsort(-so)[:k].copy())
                    top_mapped = pm.to_transformed(layer, top_orig)
                    r_orig = _neuron_restoration(self.adapter, batches, layer, top_orig)
                    r_mapped = _neuron_restoration(pm.adapter, batches, layer, top_mapped)
                    r_naive = _neuron_restoration(pm.adapter, batches, layer, top_orig)
                cmpn |= dict(
                    layer=layer, top_k=k,
                    top_channels_original=top_orig.tolist(),
                    top_channels_mapped=top_mapped.tolist(),
                    exact_patch_restoration_original=float(r_orig.mean()),
                    exact_patch_restoration_mapped=float(r_mapped.mean()),
                    exact_patch_restoration_naive_ids=float(r_naive.mean()),
                    exact_patch_max_abs_diff_mapped=float(np.abs(r_orig - r_mapped).max()),
                    naive_labels_flagged=bool(cmpn["naive_spearman"] < 0.5),
                    scores_original=so.tolist(), scores_transformed=st.tolist(),
                    scores_original_mapped=so[perms[layer].numpy()].tolist(),
                )
                neuron.append(cmpn)
                log(f"  layer {layer}: mapped rho {cmpn['mapped_spearman']:.4f}, naive rho {cmpn['naive_spearman']:.4f}, "
                    f"top-{k} restoration orig {cmpn['exact_patch_restoration_original']:.4f} "
                    f"mapped {cmpn['exact_patch_restoration_mapped']:.4f} naive {cmpn['exact_patch_restoration_naive_ids']:.4f}")
            out["neuron_level"] = neuron
            mapped_ok = all(n["mapped_spearman"] >= 0.999 and n["exact_patch_max_abs_diff_mapped"] <= 1e-3 for n in neuron)
            contradicts = cmp["classification"] == "decision_changing" or not mapped_ok
            self.findings.append(V.ChallengeFinding(
                "mlp_channel_permutation", "symmetry", True, contradicts,
                f"parity max |dlogit| {worst.max_abs_logit_diff:.3g}; block-level {cmp['classification']}; "
                f"mapped neuron attributions {'agree' if mapped_ok else 'DISAGREE'}"))
        finally:
            del pm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- counterexamples
    def evaluator(self, pred: Prediction):
        t, ref = self.spec.task, self.ref_condition
        comps = list(pred.components)
        sites = [("mlp_out", l) for l in comps]

        def fn(axes: dict) -> CX.Evaluation:
            case = ioi.render(axes, split="search")
            tok, reasons = structural_check(case, self.tokenizer, t.max_seq_len)
            if tok is None:
                return CX.Evaluation(axes, False, reasons, None, 0)
            b = make_batches([tok], 1, self.adapter.device)[0]
            with B.execution(ref):
                cl, cache = run(self.adapter, b.clean, capture=sites)
                co, _ = run(self.adapter, b.corrupt)
                mc = float(margins(cl, b.a_ids, b.b_ids)[0])
                mx = float(margins(co, b.a_ids, b.b_ids)[0])
                reasons = behavioural_check(mc, mx, t.min_contrast, t.require_corrupt_flip)
                if reasons:
                    return CX.Evaluation(axes, False, reasons, None, 2, dict(clean_margin=mc, corrupt_margin=mx))
                mp = float(joint_patch_margins(self.adapter, b, comps, clean_cache=cache)[0])
            r = (mp - mx) / (mc - mx)
            return CX.Evaluation(axes, True, [], r - pred.threshold, 3,
                                 dict(clean_margin=mc, corrupt_margin=mx, patched_margin=mp, restoration=r,
                                      clean=case.clean, corrupt=case.corrupt))
        return fn

    def counterexamples(self) -> None:
        cfg = self.spec.challenges.counterexamples
        if cfg is None or not cfg.enabled:
            return
        pred = self.claim.per_case_prediction()
        if pred is None:
            self.findings.append(V.ChallengeFinding(
                "counterfactual_inputs", "counterexamples", False, False,
                "claim has no per_case prediction, so no single input can contradict it"))
            return
        space = ioi.axis_space("discovery", self.names_ok)
        fn = self.evaluator(pred)
        log(f"counterexamples: prediction {pred.id}, budget {cfg.budget_forwards} forwards x "
            f"{len(cfg.seeds)} seeds x {cfg.strategies}")
        runs, records = [], []
        for strategy in cfg.strategies:
            for seed in cfg.seeds:
                res = CX.search(strategy, space, fn, cfg.budget_forwards, seed)
                runs.append(res.summary())
                for e in res.counterexamples:
                    records.append(CX.to_record(e) | {"strategy": strategy, "seed": seed})
        eff = []
        for strategy in cfg.strategies:
            vals = np.array([r["counterexamples_per_1k_forwards"] for r in runs if r["strategy"] == strategy])
            valid_rate = np.array([r["valid"] / max(r["proposals"], 1) for r in runs if r["strategy"] == strategy])
            lo, hi = (S.ci(np.random.default_rng(0).choice(vals, (2000, len(vals))).mean(1))
                      if len(vals) > 1 else (vals.mean(), vals.mean()))
            best = [r["best_score"] for r in runs if r["strategy"] == strategy and r["best_score"] is not None]
            eff.append(dict(strategy=strategy, seeds=len(vals), cx_per_1k_mean=float(vals.mean()),
                            cx_per_1k_ci95=[float(lo), float(hi)], valid_proposal_rate=float(valid_rate.mean()),
                            lowest_restoration_per_seed=[b + pred.threshold for b in best],
                            lowest_restoration_mean=float(np.mean(best)) + pred.threshold if best else None))
        # independent replay through a fresh evaluator (no cache)
        unique = {CX.key(r["axes"]): r for r in records}
        replayed = [CX.replay(r, fn) for r in unique.values()]
        n_repro = sum(r["reproduced"] for r in replayed)
        log(f"  {len(unique)} unique counterexamples, {n_repro} reproduce on replay")
        out = dict(prediction=pred.id, threshold=pred.threshold, components=pred.components,
                   max_violation_rate=pred.max_violation_rate, runs=runs, efficiency=eff,
                   unique_counterexamples=len(unique), reproduced=n_repro, base_input=space.base)
        base_eval = fn(space.base)
        out["base_input_evaluation"] = CX.to_record(base_eval)
        minimized, slices = [], []
        if cfg.minimize and unique and base_eval.valid and not base_eval.violates:
            worst = sorted(unique.values(), key=lambda r: (r["score"], CX.key(r["axes"])))[:5]
            seen_fixed = set()
            for r in worst:
                mz = CX.minimize(r["axes"], space.base, fn, space.is_valid)
                rec = dict(original=mz.original, minimal=mz.minimal, changed_axes=mz.changed_axes,
                           evaluation=CX.to_record(mz.evaluation), steps=mz.steps)
                minimized.append(rec)
                fk = CX.key(mz.changed_axes)
                if fk in seen_fixed or not mz.changed_axes:
                    continue
                seen_fixed.add(fk)
                sl = CX.slice_test(mz.changed_axes, space, fn, cfg.slice_cases, self.spec.seed)
                slices.append(sl)
                log(f"  minimal change {mz.changed_axes}: slice violation rate "
                    f"{sl['violations']}/{sl['valid']} CI {sl['wilson95'][0]:.2f}-{sl['wilson95'][1]:.2f}")
        elif base_eval.violates:
            out["note"] = "the base input itself violates the prediction; it is the minimal counterexample"
        out["minimized"] = minimized
        out["slices"] = slices
        self.results["counterexamples"] = out
        self.bundle.jsonl("counterexamples.jsonl", [r | {"replay": rp} for r, rp in zip(unique.values(), replayed)])
        bad_slices = [s for s in slices if s["valid"] >= 4 and s["wilson95"][0] > pred.max_violation_rate]
        contradicts = bool(bad_slices) or (pred.max_violation_rate == 0 and n_repro > 0)
        summary = (f"{len(unique)} unique counterexamples ({n_repro} reproduced); "
                   + (f"sub-scope failures: {[s['fixed'] for s in bad_slices]}" if bad_slices
                      else "no minimized counterexample generalised to a failing sub-scope"))
        self.findings.append(V.ChallengeFinding("counterfactual_inputs", "counterexamples",
                                                n_repro == len(unique), contradicts, summary))

    # ---------------------------------------------------------------- export
    def finish(self) -> Path:
        from .reports.report import write_report

        verdict = V.decide(self.claim, self.gates, self.predictions, self.findings)
        log(f"verdict: {verdict.status.value}")
        for r in verdict.reasons:
            log(f"  {r}")
        bd = self.bundle
        claim_out = self.claim.model_copy(update={"status": verdict.status})
        bd.text("claim.yaml", dump_yaml(claim_out))
        spec_out = self.spec.model_copy(update={"claim": "claim.yaml"})
        bd.text("experiment.yaml", dump_yaml(spec_out))
        bd.json("preflight.json", dict(gates={k: dict(ok=v[0], detail=v[1]) for k, v in self.gates.items()},
                                       names_excluded=self.results.get("names_excluded_not_single_token"),
                                       denominator=self.denominator(), cases=self.preflight_rows))
        raw = pd.concat(self.raw_frames, ignore_index=True)
        bd.parquet("raw_results.parquet", raw)
        bd.jsonl("interventions.jsonl", self.interventions)
        metrics = dict(verdict=verdict.to_dict(), gates={k: dict(ok=v[0], detail=v[1]) for k, v in self.gates.items()},
                       denominator=self.denominator(), **self.results)
        bd.json("metrics.json", metrics)
        manifest = dict(run_id=self.run_id, experiment_id=self.spec.experiment_id,
                        spec_sha256=self.spec_hash, source_spec=str(self.spec_path),
                        model=self.adapter.describe(), model_revision=self.spec.model_revision,
                        determinism=self.determinism, numerical_state_at_end=numerical_state(),
                        environment=environment(), seeds=dict(experiment=self.spec.seed),
                        reference_condition=self.ref_condition.model_dump(),
                        reproduce=f"kerneltrace replay {bd.root.as_posix()}")
        bd.json("manifest.json", manifest)
        write_report(bd, self.claim, metrics, manifest)
        bd.json("files.json", bd.file_hashes())
        log(f"evidence bundle: {bd.root}")
        return bd.root

    def denominator(self) -> dict:
        out = {}
        for split in ("discovery", "holdout"):
            rows = [r for r in self.preflight_rows if r["split"] == split]
            reasons: dict[str, int] = {}
            for r in rows:
                for why in (r["structural_reasons"] or []) + (r["behavioural_reasons"] or []):
                    reasons[why] = reasons.get(why, 0) + 1
            out[split] = dict(generated=len(rows), valid=sum(r["valid"] for r in rows), rejections=reasons)
        return out

    def run(self) -> Path:
        self.setup()
        self.build_tasks()
        self.baseline()
        self.numerical()
        self.symmetry()
        self.counterexamples()
        return self.finish()


def run_experiment(spec_path: str | Path, out_root: str | Path = "runs") -> Path:
    return Experiment(spec_path, out_root).run()


def replay_bundle(bundle_dir: str | Path, atol: float = 1e-5) -> dict:
    """Re-run a bundle's reference measurement from its own spec and compare per case.

    The bundle's experiment.yaml and claim.yaml are used, so the replay needs
    nothing outside the bundle except the code and the pinned model weights.
    """
    bundle_dir = Path(bundle_dir)
    exp = Experiment(bundle_dir / "experiment.yaml", out_root=bundle_dir / "replays")
    exp.setup()
    exp.build_tasks()
    ref = exp.ref_condition
    meas = measure(exp.adapter, exp.cases, exp.spec.task.batch_size, ref, exp.claim_sets(),
                   identity_controls=False, keep_logits=False)
    raw = pd.read_parquet(bundle_dir / "raw_results.parquet")
    old = raw[raw.condition == ref.name].set_index(["case_id", "layer"])
    new = exp.raw_frame(meas, np.ones(len(exp.cases), bool)).set_index(["case_id", "layer"])
    missing = sorted(set(old.index.get_level_values(0)) ^ set(new.index.get_level_values(0)))
    joined = old.join(new, rsuffix="_replay", how="inner")
    diffs = {col: float(np.abs(joined[col] - joined[f"{col}_replay"]).max())
             for col in ("clean_margin", "corrupt_margin", "patched_margin")}
    old_manifest = __import__("json").loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    env_now = environment()
    env_then = old_manifest["environment"]
    changed = {k: (env_then.get(k), env_now.get(k)) for k in
               ("torch", "torch_cuda", "transformers", "gpu_name", "nvidia_driver", "git_commit")
               if env_then.get(k) != env_now.get(k)}
    out = dict(bundle=str(bundle_dir), condition=ref.name, compared_rows=len(joined),
               case_sets_differ=missing, max_abs_diff=diffs, atol=atol,
               reproduced=not missing and all(v <= atol for v in diffs.values()),
               bitwise=not missing and all(v == 0 for v in diffs.values()),
               environment_changes=changed)
    exp.bundle.json("replay.json", out)
    log(f"replay {'REPRODUCED' if out['reproduced'] else 'MISMATCH'}: {diffs}")
    return out
