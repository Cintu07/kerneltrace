"""The search engine against a synthetic claim with a planted failure (FR-06).

The oracle 'model' satisfies the claim everywhere except when template 3 is
combined with the object "kiss". A correct engine finds it, minimises it to
exactly those two axis changes, and confirms the slice fails.
"""
import pytest

from kerneltrace.search import counterexample as CX
from kerneltrace.tasks import ioi

SPACE = ioi.axis_space("discovery")


def oracle(axes):
    if axes["io"] == "John":  # an invalid region: the 'model' fails the task here
        return CX.Evaluation(axes, False, ["clean_does_not_prefer_answer"], None, 2)
    planted = axes["template"] == 3 and axes["object"] == "kiss"
    score = -0.4 if planted else 0.3 + 0.01 * axes["template"]
    return CX.Evaluation(axes, True, [], score, 3)


@pytest.mark.parametrize("strategy", ["random", "coverage", "guided"])
def test_each_strategy_finds_planted_failure(strategy):
    res = CX.search(strategy, SPACE, oracle, budget=900, seed=0)
    assert res.forwards <= 900 + 3
    assert res.counterexamples, res.summary()
    for e in res.counterexamples:
        assert e.axes["template"] == 3 and e.axes["object"] == "kiss"


def test_invalid_inputs_are_never_counterexamples():
    res = CX.search("random", SPACE, oracle, budget=600, seed=1)
    assert all(e.valid for e in res.counterexamples)
    assert any(not e.valid for e in res.evaluated)


def test_cache_charges_each_input_once():
    calls = []

    def counting(axes):
        calls.append(axes)
        return oracle(axes)

    ev = CX.CachedEvaluator(counting)
    ev(SPACE.base)
    ev(SPACE.base)
    assert len(calls) == 1 and ev.forwards == 3


def test_minimize_finds_exactly_the_planted_axes():
    cx = dict(template=3, order="S_first", io="Sarah", s="Tom", place="beach", object="kiss")
    mz = CX.minimize(cx, SPACE.base, oracle, SPACE.is_valid)
    assert mz.changed_axes == {"template": 3, "object": "kiss"}
    assert mz.evaluation.violates


def test_minimize_is_deterministic():
    cx = dict(template=3, order="S_first", io="Sarah", s="Tom", place="beach", object="kiss")
    a = CX.minimize(cx, SPACE.base, oracle, SPACE.is_valid)
    b = CX.minimize(cx, SPACE.base, oracle, SPACE.is_valid)
    assert a.minimal == b.minimal and a.steps == b.steps


def test_minimize_rejects_non_counterexample():
    with pytest.raises(ValueError):
        CX.minimize(SPACE.base, SPACE.base, oracle)


def test_slice_generalises_the_minimal_change():
    s = CX.slice_test({"template": 3, "object": "kiss"}, SPACE, oracle, n=20, seed=0)
    assert s["valid"] == 20 and s["violations"] == 20
    s = CX.slice_test({"template": 3}, SPACE, oracle, n=40, seed=0)
    assert 0 < s["violations"] < 40


def test_replay_reproduces_and_detects_drift():
    e = oracle(dict(template=3, order="IO_first", io="Mary", s="John", place="store", object="kiss"))
    rec = CX.to_record(e)
    assert CX.replay(rec, oracle)["reproduced"]
    drifted = dict(rec, score=rec["score"] + 1.0)
    assert not CX.replay(drifted, oracle)["reproduced"]
