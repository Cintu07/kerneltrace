"""capital cities: factual recall with a country-swap corruption.

clean:    The capital of France is the city of    -> Paris
corrupt:  The capital of Germany is the city of   -> Berlin

the metric is logit(clean capital) - logit(corrupt capital) at the last position. the two prompts
differ in one token, so positions line up. ioi is mostly done by attention heads; recall like this
leans on mid-layer mlps, which is why it is the second task.
"""
from __future__ import annotations

import random

from .ioi import IOICase, case_id

TEMPLATES = [
    "The capital of {C} is the city of",
    "The capital city of {C} is",
    "In {C}, the seat of government is the city of",
    "Q: What is the capital of {C}? A: The capital is",
]
CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo",
    "China": "Beijing", "Russia": "Moscow", "Egypt": "Cairo", "Greece": "Athens", "England": "London",
    "Ireland": "Dublin", "Austria": "Vienna", "Poland": "Warsaw", "Canada": "Ottawa", "Australia": "Canberra",
    "Turkey": "Ankara", "Iran": "Tehran", "Iraq": "Baghdad", "Thailand": "Bangkok", "Kenya": "Nairobi",
    "Peru": "Lima", "Cuba": "Havana", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Portugal": "Lisbon", "Hungary": "Budapest", "Belgium": "Brussels",
    "Korea": "Seoul", "Vietnam": "Hanoi", "Afghanistan": "Kabul", "Syria": "Damascus", "Lebanon": "Beirut",
    "Chile": "Santiago", "Mexico": "Mexico", "Indonesia": "Jakarta", "Philippines": "Manila",
    "Ukraine": "Kyiv", "Scotland": "Edinburgh", "Wales": "Cardiff", "Switzerland": "Bern",
}


def render(template: int, country: str, other: str) -> IOICase:
    axes = {"task": "capitals", "template": template, "country": country, "other": other}
    t = TEMPLATES[template]
    return IOICase(case_id=case_id(axes), axes=axes, clean=t.format(C=country), corrupt=t.format(C=other),
                   answer=" " + CAPITALS[country], distractor=" " + CAPITALS[other], split="capitals",
                   template=template)


def generate(n: int, seed: int = 7) -> list[IOICase]:
    """every (template, country, other country) triple, in a fixed shuffled order. n caps the pool."""
    pool = [render(t, a, b) for t in range(len(TEMPLATES)) for a in CAPITALS for b in CAPITALS
            if a != b and CAPITALS[a] != CAPITALS[b] and CAPITALS[a] != a and CAPITALS[b] != b]
    random.Random(f"capitals:{seed}").shuffle(pool)
    return pool[:n]
