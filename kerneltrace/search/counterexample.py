"""Counterexample search over a discrete space of task-valid inputs.

The engine is model-agnostic. It needs an axis space and an evaluator that
maps one point to an Evaluation: whether the input is a valid test of the
task, and a score that is negative when the claim's per-case prediction fails.
A counterexample is a valid input with a negative score.

Cost is counted in model forward passes, so strategies are compared at equal
compute rather than equal numbers of proposals.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from ..metrics.stats import wilson


@dataclass
class Evaluation:
    axes: dict
    valid: bool
    reasons: list[str]
    score: float | None  # claim margin: negative means the prediction failed
    forwards: int
    detail: dict = field(default_factory=dict)

    @property
    def violates(self) -> bool:
        return self.valid and self.score is not None and self.score < 0


class Space(Protocol):
    values: dict[str, list]
    base: dict

    def sample(self, rng: random.Random) -> dict: ...
    def is_valid(self, axes: dict) -> bool: ...


def key(axes: dict) -> str:
    return json.dumps(axes, sort_keys=True)


class CachedEvaluator:
    """Memoises evaluations and charges forwards only for new inputs."""

    def __init__(self, fn: Callable[[dict], Evaluation]):
        self.fn = fn
        self.cache: dict[str, Evaluation] = {}
        self.forwards = 0

    def __call__(self, axes: dict) -> Evaluation:
        k = key(axes)
        if k not in self.cache:
            ev = self.fn(axes)
            self.forwards += ev.forwards
            self.cache[k] = ev
        return self.cache[k]


@dataclass
class SearchResult:
    strategy: str
    seed: int
    budget: int
    forwards: int
    evaluated: list[Evaluation]

    @property
    def valid(self) -> list[Evaluation]:
        return [e for e in self.evaluated if e.valid]

    @property
    def counterexamples(self) -> list[Evaluation]:
        return [e for e in self.evaluated if e.violates]

    def summary(self) -> dict:
        n_cx = len(self.counterexamples)
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "budget_forwards": self.budget,
            "forwards_used": self.forwards,
            "proposals": len(self.evaluated),
            "valid": len(self.valid),
            "counterexamples": n_cx,
            "counterexamples_per_1k_forwards": 1000 * n_cx / max(self.forwards, 1),
            "best_score": min((e.score for e in self.valid), default=None),
        }


def _random(space: Space, ev: CachedEvaluator, budget: int, rng: random.Random, out: list):
    stall = 0
    while ev.forwards < budget and stall < 1000:
        before = len(ev.cache)
        out.append(ev(space.sample(rng)))
        stall = stall + 1 if len(ev.cache) == before else 0


def _coverage(space: Space, ev: CachedEvaluator, budget: int, rng: random.Random, out: list):
    """Cycle every axis through all its values so rare values are tested early.

    Each proposal takes the least-used value on one axis (rotating which axis)
    and draws the rest at random. This guarantees every value of every axis is
    visited within sum(len(values)) proposals.
    """
    counts = {a: {json.dumps(v): 0 for v in vals} for a, vals in space.values.items()}
    axes_order = list(space.values)
    i, stall = 0, 0
    while ev.forwards < budget and stall < 1000:
        axis = axes_order[i % len(axes_order)]
        i += 1
        fewest = min(counts[axis].values())
        choices = [v for v in space.values[axis] if counts[axis][json.dumps(v)] == fewest]
        proposal = space.sample(rng)
        proposal[axis] = rng.choice(choices)
        if not space.is_valid(proposal):
            stall += 1
            continue
        before = len(ev.cache)
        out.append(ev(proposal))
        stall = stall + 1 if len(ev.cache) == before else 0
        for a, v in proposal.items():
            counts[a][json.dumps(v)] += 1


def _mutate(axes: dict, space: Space, rng: random.Random) -> dict:
    for _ in range(100):
        new = dict(axes)
        axis = rng.choice(list(space.values))
        new[axis] = rng.choice(space.values[axis])
        if new != axes and space.is_valid(new):
            return new
    return space.sample(rng)


def _guided(space: Space, ev: CachedEvaluator, budget: int, rng: random.Random, out: list,
            n_seed: int = 8, patience: int = 6):
    """Hill-climb on the claim margin: mutate one axis at a time towards lower score.

    Starts from a few random valid points, restarts from a fresh random point
    after `patience` non-improving steps, and never re-counts a cached input.
    """
    def fresh():
        while ev.forwards < budget:
            e = ev(space.sample(rng))
            out.append(e)
            if e.valid:
                return e
        return None

    pool = [e for e in (fresh() for _ in range(n_seed)) if e is not None]
    if not pool:
        return
    current = min(pool, key=lambda e: e.score)
    stale, stall = 0, 0
    while ev.forwards < budget and stall < 1000:
        before = len(ev.cache)
        cand = ev(_mutate(current.axes, space, rng))
        if len(ev.cache) == before:
            stall += 1
            continue
        stall = 0
        out.append(cand)
        if cand.valid and cand.score < current.score:
            current, stale = cand, 0
        else:
            stale += 1
        if stale >= patience:
            restart = fresh()
            if restart is None:
                break
            current, stale = restart, 0


STRATEGIES = {"random": _random, "coverage": _coverage, "guided": _guided}


def search(strategy: str, space: Space, fn: Callable[[dict], Evaluation], budget: int,
           seed: int) -> SearchResult:
    ev = CachedEvaluator(fn)
    rng = random.Random(f"{strategy}:{seed}")
    evaluated: list[Evaluation] = []
    STRATEGIES[strategy](space, ev, budget, rng, evaluated)
    seen, unique = set(), []
    for e in evaluated:
        k = key(e.axes)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return SearchResult(strategy, seed, budget, ev.forwards, unique)


@dataclass
class Minimized:
    original: dict
    minimal: dict
    changed_axes: dict  # axis -> value that differs from the base input
    evaluation: Evaluation
    steps: list[dict]


def minimize(cx: dict, base: dict, fn: Callable[[dict], Evaluation],
             is_valid: Callable[[dict], bool] | None = None) -> Minimized:
    """Revert axes to the base input while the input stays valid and still fails.

    Deterministic: axes are tried in sorted order, repeated to a fixpoint. The
    result differs from the base on as few axes as this greedy pass can manage
    (1-minimal: reverting any single remaining axis breaks validity or the failure).
    """
    current = dict(cx)
    first = fn(current)
    if not first.violates:
        raise ValueError("minimize() needs a valid counterexample to start from")
    last = first
    steps = []
    changed = True
    while changed:
        changed = False
        for axis in sorted(current):
            if current[axis] == base[axis]:
                continue
            trial = dict(current)
            trial[axis] = base[axis]
            if is_valid is not None and not is_valid(trial):
                steps.append({"axis": axis, "kept": False, "why": "outside_axis_space"})
                continue
            e = fn(trial)
            ok = e.violates
            steps.append({"axis": axis, "reverted_to": base[axis], "kept": ok,
                          "valid": e.valid, "score": e.score})
            if ok:
                current, last, changed = trial, e, True
    diff = {a: v for a, v in current.items() if v != base[a]}
    return Minimized(cx, current, diff, last, steps)


def slice_test(fixed: dict, space: Space, fn: Callable[[dict], Evaluation], n: int,
               seed: int, max_proposals: int | None = None) -> dict:
    """Hold some axes fixed, draw the rest at random, and measure the failure rate.

    This turns one minimal counterexample into a statement about a sub-scope:
    "with these axis values the prediction fails on k of n valid inputs".
    """
    rng = random.Random(f"slice:{seed}:{key(fixed)}")
    valid, viol, proposals, forwards = 0, 0, 0, 0
    seen = set()
    cap = max_proposals or 20 * n
    while valid < n and proposals < cap:
        proposals += 1
        axes = space.sample(rng) | fixed
        if not space.is_valid(axes) or key(axes) in seen:
            continue
        seen.add(key(axes))
        e = fn(axes)
        forwards += e.forwards
        if e.valid:
            valid += 1
            viol += e.violates
    lo, hi = wilson(viol, valid)
    return {"fixed": fixed, "valid": valid, "violations": viol,
            "violation_rate": viol / valid if valid else None,
            "wilson95": [lo, hi], "forwards": forwards}


def replay(record: dict, fn: Callable[[dict], Evaluation], tol: float = 1e-3) -> dict:
    """Re-evaluate a saved counterexample and check it reproduces."""
    e = fn(record["axes"])
    same = (e.valid == record["valid"] and e.violates == record["violates"]
            and (e.score is None or abs(e.score - record["score"]) <= tol))
    return {"axes": record["axes"], "reproduced": same, "score": e.score,
            "recorded_score": record["score"]}


def to_record(e: Evaluation) -> dict:
    d = asdict(e)
    d["violates"] = e.violates
    return d
