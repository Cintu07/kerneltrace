"""from measurements to a claim status.

the rules run in this order:

1. a failed gate (too few valid cases, a control patch that did not come back
   null) makes the claim inconclusive. nothing later can rescue it.
2. a prediction whose interval sits on the wrong side of its threshold makes
   the claim falsified_in_scope.
3. a prediction whose interval straddles its threshold makes it inconclusive.
4. an invariance the claim lists that no valid challenge tested makes it
   inconclusive. an untested invariance is a hole in the evidence.
5. a valid challenge that contradicts the claim makes it challenged.
6. otherwise the claim is supported_in_scope, and only in that scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..metrics.stats import ci, cluster_bootstrap, summarize, wilson
from .spec import ClaimCard, Prediction, Status


@dataclass
class SplitData:
    """what the verdict needs from one split, valid cases only."""

    split: str
    clusters: np.ndarray          # [N] template id per case
    effects: np.ndarray           # [N, L] patched - corrupt per layer
    joint_restoration: dict       # components tuple -> [N]
    joint_effect: dict            # components tuple -> [N]

    @property
    def n(self) -> int:
        return len(self.clusters)


@dataclass
class PredictionResult:
    id: str
    kind: str
    split: str
    outcome: str  # pass | fail | inconclusive
    reason: str
    stats: dict = field(default_factory=dict)


def evaluate_prediction(p: Prediction, data: SplitData, n_boot: int, seed: int,
                        min_valid: int) -> PredictionResult:
    def result(outcome, reason, **stats):
        return PredictionResult(p.id, p.kind, p.split, outcome, reason, stats)

    if data.n < min_valid:
        return result("inconclusive", f"only {data.n} valid cases, need {min_valid}")
    comps = tuple(p.components)
    # bf16 logit spacing can squash a case's contrast to exactly zero, which
    # leaves restoration undefined. nan compares false, so it would slip through as a pass.
    values = {"restoration_at_least": data.joint_restoration.get(comps),
              "effect_positive": data.joint_effect.get(comps),
              "in_top_k": data.effects}[p.kind]
    bad = int((~np.isfinite(values)).sum())
    if bad:
        return result("inconclusive", f"{bad} values are undefined (zero clean-corrupt contrast under "
                                      f"this condition)", n=data.n, undefined=bad)
    if p.kind == "restoration_at_least" and p.quantifier == "mean":
        s = summarize(data.joint_restoration[comps], data.clusters, n_boot, seed)
        if s["ci95_low"] >= p.threshold:
            return result("pass", f"mean restoration CI [{s['ci95_low']:.3f}, {s['ci95_high']:.3f}] >= {p.threshold}", **s)
        if s["ci95_high"] < p.threshold:
            return result("fail", f"mean restoration CI [{s['ci95_low']:.3f}, {s['ci95_high']:.3f}] < {p.threshold}", **s)
        return result("inconclusive", f"mean restoration CI [{s['ci95_low']:.3f}, {s['ci95_high']:.3f}] straddles {p.threshold}", **s)
    if p.kind == "restoration_at_least":
        r = data.joint_restoration[comps]
        k = int((r < p.threshold).sum())
        lo, hi = wilson(k, data.n)
        stats = dict(n=data.n, violations=k, violation_rate=k / data.n, wilson95_low=lo, wilson95_high=hi)
        if hi <= p.max_violation_rate:
            return result("pass", f"{k}/{data.n} cases below {p.threshold}; rate CI high {hi:.3f} <= {p.max_violation_rate}", **stats)
        if lo > p.max_violation_rate:
            return result("fail", f"{k}/{data.n} cases below {p.threshold}; rate CI low {lo:.3f} > {p.max_violation_rate}", **stats)
        return result("inconclusive", f"{k}/{data.n} cases below {p.threshold}; rate CI [{lo:.3f}, {hi:.3f}] straddles {p.max_violation_rate}", **stats)
    if p.kind == "effect_positive":
        s = summarize(data.joint_effect[comps], data.clusters, n_boot, seed)
        if s["ci95_low"] > 0:
            return result("pass", f"mean effect CI low {s['ci95_low']:.4f} > 0", **s)
        if s["ci95_high"] < 0:
            return result("fail", f"mean effect CI high {s['ci95_high']:.4f} < 0", **s)
        return result("inconclusive", "mean effect CI includes 0", **s)
    if p.kind == "in_top_k":
        boot = cluster_bootstrap(data.effects, data.clusters, n_boot, seed)  # [B, L]
        order = np.argsort(-boot, axis=1)[:, :p.k]
        hit = np.array([set(comps) <= set(row.tolist()) for row in order])
        frac = float(hit.mean())
        point = np.argsort(-data.effects.mean(0))[:p.k].tolist()
        stats = dict(n=data.n, bootstrap_fraction_in_top_k=frac, point_top_k=point)
        if frac >= 0.95:
            return result("pass", f"components in top-{p.k} in {frac:.1%} of resamples", **stats)
        if frac <= 0.05:
            return result("fail", f"components in top-{p.k} in only {frac:.1%} of resamples", **stats)
        return result("inconclusive", f"components in top-{p.k} in {frac:.1%} of resamples", **stats)
    raise ValueError(f"unhandled prediction kind {p.kind}")


@dataclass
class ChallengeFinding:
    invariance: str      # matches Invariance.kind, or "out_of_scope"
    name: str
    valid: bool          # the challenge's own prerequisites passed
    contradicts: bool
    summary: str


@dataclass
class Verdict:
    status: Status
    reasons: list[str]
    predictions: list[PredictionResult]
    findings: list[ChallengeFinding]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "predictions": [asdict(p) for p in self.predictions],
            "findings": [asdict(f) for f in self.findings],
        }


def decide(claim: ClaimCard, gates: dict[str, tuple[bool, str]],
           predictions: list[PredictionResult], findings: list[ChallengeFinding]) -> Verdict:
    reasons = [f"gate {name} failed: {why}" for name, (ok, why) in gates.items() if not ok]
    if reasons:
        return Verdict(Status.INCONCLUSIVE, reasons, predictions, findings)
    failed = [p for p in predictions if p.outcome == "fail"]
    if failed:
        return Verdict(Status.FALSIFIED_IN_SCOPE,
                       [f"prediction {p.id} failed: {p.reason}" for p in failed],
                       predictions, findings)
    unclear = [p for p in predictions if p.outcome == "inconclusive"]
    if unclear:
        return Verdict(Status.INCONCLUSIVE,
                       [f"prediction {p.id} inconclusive: {p.reason}" for p in unclear],
                       predictions, findings)
    required = {inv.kind for inv in claim.invariances}
    tested = {f.invariance for f in findings if f.valid}
    untested = sorted(required - tested)
    if untested:
        return Verdict(Status.INCONCLUSIVE,
                       [f"invariance {k} listed in the claim was not validly tested" for k in untested],
                       predictions, findings)
    contra = [f for f in findings if f.valid and f.contradicts and f.invariance in required]
    if contra:
        return Verdict(Status.CHALLENGED,
                       [f"{f.name} ({f.invariance}): {f.summary}" for f in contra],
                       predictions, findings)
    return Verdict(Status.SUPPORTED_IN_SCOPE,
                   ["all predictions passed and every listed invariance was tested without contradiction"],
                   predictions, findings)
