"""Indirect object identification (IOI) with an S2-flip corruption.

clean:    When Mary and John went to the store, John gave a drink to   -> Mary
corrupt:  When Mary and John went to the store, Mary gave a drink to   -> John

Clean and corrupt differ in exactly one token (the second subject mention), so
token positions line up and a full-sequence patch is well defined. The metric
is logit(IO) - logit(S) at the last position, measured on both runs.

Every case is a point in a small discrete space of axes (template, name order,
names, place, object). Discovery and holdout use disjoint templates and
disjoint names, fixed here rather than chosen after seeing results.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

TEMPLATES = [
    "When {N1} and {N2} went to the {PLACE}, {S2} gave a {OBJECT} to",
    "When {N1} and {N2} got a {OBJECT} at the {PLACE}, {S2} decided to give it to",
    "Then, {N1} and {N2} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "Then, {N1} and {N2} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to",
    "While {N1} and {N2} were working at the {PLACE}, {S2} gave a {OBJECT} to",
    "After {N1} and {N2} went to the {PLACE}, {S2} gave a {OBJECT} to",
    "Friends {N1} and {N2} found a {OBJECT} at the {PLACE}. {S2} gave it to",
    "When {N1} and {N2} arrived at the {PLACE}, {S2} handed the {OBJECT} to",
    # holdout templates: never used for discovery or counterexample search
    "{N1} and {N2} were at the {PLACE}. {S2} passed the {OBJECT} to",
    "After lunch, {N1} and {N2} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "While spending time together, {N1} and {N2} found a {OBJECT} at the {PLACE}. {S2} gave it to",
    "The {PLACE} was busy when {N1} and {N2} arrived. {S2} brought a {OBJECT} to",
    "Yesterday {N1} and {N2} walked to the {PLACE}, and {S2} gave a {OBJECT} to",
    "Once {N1} and {N2} reached the {PLACE}, {S2} offered a {OBJECT} to",
    "{N1} and {N2} met at the {PLACE}. Later, {S2} sent a {OBJECT} to",
    "During a visit to the {PLACE}, {N1} and {N2} talked, and {S2} gave a {OBJECT} to",
]
HOLDOUT_TEMPLATES = frozenset(range(8, 16))

NAMES = [
    "Mary", "John", "Alice", "Bob", "Sarah", "David", "Emily", "James",
    "Tom", "Anna", "Michael", "Lisa", "Paul", "Kate", "Mark", "Laura",
    "Daniel", "Rachel", "Steven", "Jessica", "Robert", "Amy", "Kevin", "Linda",
    "Peter", "Susan", "Brian", "Karen", "Richard", "Nancy", "Eric", "Helen",
    "Andrew", "Sophie", "Jason", "Grace", "Frank", "Emma", "George", "Lucy",
]
PLACES = ["store", "park", "school", "hospital", "office", "station",
          "restaurant", "garden", "library", "beach", "market", "house"]
OBJECTS = ["drink", "book", "ring", "bone", "kiss", "basketball",
           "computer", "necklace", "snack", "letter"]
ORDERS = ["IO_first", "S_first"]
AXES = ("template", "order", "io", "s", "place", "object")


def _holdout_name(i: int) -> bool:
    return i % 4 == 3


DISCOVERY_NAMES = [n for i, n in enumerate(NAMES) if not _holdout_name(i)]
HOLDOUT_NAMES = [n for i, n in enumerate(NAMES) if _holdout_name(i)]


@dataclass(frozen=True)
class IOICase:
    case_id: str
    axes: dict = field(hash=False)
    clean: str
    corrupt: str
    answer: str
    distractor: str
    split: str
    template: int


def case_id(axes: dict) -> str:
    return hashlib.sha256(json.dumps(axes, sort_keys=True).encode()).hexdigest()[:12]


def render(axes: dict, split: str | None = None) -> IOICase:
    """Turn one point in the axis space into a clean/corrupt prompt pair."""
    io, s = axes["io"], axes["s"]
    if io == s:
        raise ValueError("indirect object and subject must differ")
    n1, n2 = (io, s) if axes["order"] == "IO_first" else (s, io)
    template = TEMPLATES[axes["template"]]
    fill = dict(N1=n1, N2=n2, PLACE=axes["place"], OBJECT=axes["object"])
    if split is None:
        split = "holdout" if axes["template"] in HOLDOUT_TEMPLATES else "discovery"
    return IOICase(
        case_id=case_id(axes),
        axes=dict(axes),
        clean=template.format(S2=s, **fill),
        corrupt=template.format(S2=io, **fill),
        answer=" " + io,
        distractor=" " + s,
        split=split,
        template=axes["template"],
    )


@dataclass
class AxisSpace:
    """The discrete input space a split (or a search) is allowed to draw from."""

    values: dict[str, list]
    base: dict

    def sample(self, rng: random.Random) -> dict:
        while True:
            axes = {k: rng.choice(v) for k, v in self.values.items()}
            if axes["io"] != axes["s"]:
                return axes

    def is_valid(self, axes: dict) -> bool:
        return (axes["io"] != axes["s"]
                and all(axes[k] in v for k, v in self.values.items()))


def axis_space(split: str, names_ok=None) -> AxisSpace:
    """Axis values for a split. `names_ok` filters names, e.g. single-token only."""
    if split == "discovery":
        templates = [t for t in range(len(TEMPLATES)) if t not in HOLDOUT_TEMPLATES]
        names = DISCOVERY_NAMES
    elif split == "holdout":
        templates = sorted(HOLDOUT_TEMPLATES)
        names = HOLDOUT_NAMES
    else:
        raise ValueError(f"unknown split {split!r}")
    if names_ok is not None:
        names = [n for n in names if names_ok(n)]
    if len(names) < 2:
        raise ValueError(f"fewer than two usable names in the {split} split")
    values = dict(template=templates, order=ORDERS, io=names, s=names,
                  place=PLACES, object=OBJECTS)
    base = dict(template=templates[0], order=ORDERS[0], io=names[0], s=names[1],
                place=PLACES[0], object=OBJECTS[0])
    return AxisSpace(values=values, base=base)


def generate(split: str, n: int, seed: int, names_ok=None) -> list[IOICase]:
    """Sample `n` distinct cases from a split, deterministically from `seed`."""
    space = axis_space(split, names_ok)
    rng = random.Random(f"ioi:{split}:{seed}")
    seen: dict[str, IOICase] = {}
    attempts = 0
    while len(seen) < n:
        attempts += 1
        if attempts > 100 * n:
            raise RuntimeError(f"could not draw {n} distinct {split} cases")
        case = render(space.sample(rng), split)
        seen.setdefault(case.case_id, case)
    return list(seen.values())
