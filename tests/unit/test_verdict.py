"""The status rules. A failed prerequisite must never yield a passing status."""
import numpy as np

from kerneltrace.core import verdict as V
from kerneltrace.core.spec import ClaimCard, Prediction, Status


def claim(invariances=()):
    return ClaimCard(
        claim_id="t", hypothesis="h", mechanism="m", model="m", model_revision="r", unit="u",
        scope=dict(input_distribution="x"),
        predictions=[dict(id="p", kind="restoration_at_least", components=[0], threshold=0.5)],
        invariances=[dict(kind=k) for k in invariances],
        falsification=["f"],
    )


def pr(outcome):
    return V.PredictionResult("p", "restoration_at_least", "holdout", outcome, "r")


OK = {"g": (True, "ok")}


def test_failed_gate_overrides_passing_predictions():
    v = V.decide(claim(), {"g": (False, "bad")}, [pr("pass")], [])
    assert v.status == Status.INCONCLUSIVE


def test_failed_prediction_falsifies():
    assert V.decide(claim(), OK, [pr("pass"), pr("fail")], []).status == Status.FALSIFIED_IN_SCOPE


def test_unclear_prediction_is_inconclusive():
    assert V.decide(claim(), OK, [pr("inconclusive")], []).status == Status.INCONCLUSIVE


def test_untested_invariance_blocks_support():
    c = claim(["attention_backend"])
    assert V.decide(c, OK, [pr("pass")], []).status == Status.INCONCLUSIVE
    invalid = V.ChallengeFinding("attention_backend", "n", valid=False, contradicts=False, summary="")
    assert V.decide(c, OK, [pr("pass")], [invalid]).status == Status.INCONCLUSIVE


def test_valid_contradiction_challenges():
    c = claim(["attention_backend"])
    f = V.ChallengeFinding("attention_backend", "n", valid=True, contradicts=True, summary="flip")
    assert V.decide(c, OK, [pr("pass")], [f]).status == Status.CHALLENGED


def test_out_of_scope_contradiction_does_not_challenge():
    c = claim(["attention_backend"])
    ok = V.ChallengeFinding("attention_backend", "a", True, False, "")
    oos = V.ChallengeFinding("out_of_scope", "bf16", True, True, "flip under bf16")
    assert V.decide(c, OK, [pr("pass")], [ok, oos]).status == Status.SUPPORTED_IN_SCOPE


def _data(rest, clusters=None):
    rest = np.asarray(rest, float)
    clusters = np.arange(len(rest)) % 4 if clusters is None else clusters
    return V.SplitData("holdout", clusters, np.zeros((len(rest), 2)), {(0,): rest}, {(0,): rest})


def test_mean_prediction_uses_interval():
    p = Prediction(id="p", kind="restoration_at_least", components=[0], threshold=0.5)
    assert V.evaluate_prediction(p, _data([0.9] * 40), 500, 0, 10).outcome == "pass"
    assert V.evaluate_prediction(p, _data([0.1] * 40), 500, 0, 10).outcome == "fail"
    mixed = [0.0] * 20 + [1.0] * 20  # two templates restore nothing, two restore fully
    assert V.evaluate_prediction(p, _data(mixed, np.repeat(np.arange(4), 10)), 500, 0, 10).outcome == "inconclusive"


def test_too_few_cases_is_inconclusive_not_pass():
    p = Prediction(id="p", kind="restoration_at_least", components=[0], threshold=0.5)
    assert V.evaluate_prediction(p, _data([0.9] * 5), 500, 0, 10).outcome == "inconclusive"


def test_per_case_prediction():
    p = Prediction(id="p", kind="restoration_at_least", components=[0], threshold=0.5,
                   quantifier="per_case", max_violation_rate=0.1)
    assert V.evaluate_prediction(p, _data([0.9] * 100), 500, 0, 10).outcome == "pass"
    assert V.evaluate_prediction(p, _data([0.9] * 50 + [0.1] * 50), 500, 0, 10).outcome == "fail"


def test_in_top_k():
    p = Prediction(id="p", kind="in_top_k", components=[1], k=1)
    eff = np.tile([0.1, 2.0], (40, 1))
    d = V.SplitData("holdout", np.arange(40) % 4, eff, {}, {})
    assert V.evaluate_prediction(p, d, 500, 0, 10).outcome == "pass"
    p0 = Prediction(id="p", kind="in_top_k", components=[0], k=1)
    assert V.evaluate_prediction(p0, d, 500, 0, 10).outcome == "fail"
