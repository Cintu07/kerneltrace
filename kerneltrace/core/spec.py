"""Claim cards and experiment specs.

A claim card states what a mechanism is supposed to do and, more importantly,
which measurements would count against it. An experiment spec says how the
claim is tested. Both are plain YAML validated here, so a malformed claim is
rejected before any GPU time is spent on it.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(str, Enum):
    NOT_TESTED = "not_tested"
    SUPPORTED_IN_SCOPE = "supported_in_scope"
    CHALLENGED = "challenged"
    FALSIFIED_IN_SCOPE = "falsified_in_scope"
    INCONCLUSIVE = "inconclusive"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Prediction(Strict):
    """One observable consequence of the hypothesis.

    kind:
      restoration_at_least  patching `components` jointly (clean into corrupt)
                            restores at least `threshold` of the logit contrast.
                            quantifier=mean tests the mean with a bootstrap CI;
                            quantifier=per_case requires each case to clear the
                            threshold, allowing `max_violation_rate` failures.
      effect_positive       the joint patch moves the margin towards clean.
      in_top_k              every listed component ranks in the top `k` layers
                            by mean causal effect.
    """

    id: str
    kind: Literal["restoration_at_least", "effect_positive", "in_top_k"]
    components: list[int] = Field(min_length=1)
    threshold: float = 0.0
    quantifier: Literal["mean", "per_case"] = "mean"
    max_violation_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    k: int = Field(default=3, ge=1)
    split: Literal["discovery", "holdout"] = "holdout"


class Invariance(Strict):
    kind: Literal[
        "attention_backend",
        "batching",
        "repeat",
        "tf32",
        "mlp_channel_permutation",
        "counterfactual_inputs",
    ]
    note: str = ""


class Scope(Strict):
    task: str = "ioi_s2_flip"
    input_distribution: str
    backends: list[str] = ["eager", "sdpa"]
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"


class ClaimCard(Strict):
    claim_id: str
    hypothesis: str
    mechanism: str
    model: str
    model_revision: str
    unit: str
    metric: str = "answer_minus_distractor_logit_margin"
    intervention: str = "patch_clean_activation_into_corrupt_run"
    scope: Scope
    predictions: list[Prediction] = Field(min_length=1)
    invariances: list[Invariance] = []
    falsification: list[str] = Field(min_length=1)
    status: Status = Status.NOT_TESTED

    @model_validator(mode="after")
    def _unique_prediction_ids(self):
        ids = [p.id for p in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate prediction ids: {ids}")
        return self

    def per_case_prediction(self) -> Prediction | None:
        """The prediction a single input can contradict, used by counterexample search."""
        for p in self.predictions:
            if p.kind == "restoration_at_least" and p.quantifier == "per_case":
                return p
        return None


class TaskSpec(Strict):
    name: Literal["ioi_s2_flip"] = "ioi_s2_flip"
    n_discovery: int = Field(default=96, ge=4)
    n_holdout: int = Field(default=96, ge=4)
    min_contrast: float = Field(default=1.0, gt=0.0)
    require_corrupt_flip: bool = True
    min_valid: int = Field(default=20, ge=2)
    batch_size: int = Field(default=16, ge=1)
    max_seq_len: int = Field(default=64, ge=8)


class BaselineSpec(Strict):
    identity_controls: bool = True
    random_circuit_controls: int = Field(default=20, ge=0)
    bootstrap: int = Field(default=2000, ge=100)
    fidelity_top_k: int = Field(default=3, ge=1)


class ConditionSpec(Strict):
    name: str
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    sdpa_kernel: Literal["math", "efficient", "flash", "cudnn"] | None = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int | None = None
    tf32: bool = False


class NumericalSpec(Strict):
    enabled: bool = True
    reference: ConditionSpec = ConditionSpec(name="eager_fp32")
    conditions: list[ConditionSpec] = []
    effect_atol: float = 1e-4
    effect_rtol: float = 1e-3
    allow_multi_factor: bool = False


class SymmetrySpec(Strict):
    enabled: bool = True
    seed: int = 1
    parity_atol: float = 1e-3
    neuron_layers: list[int] = []
    top_k_neurons: int = Field(default=20, ge=1)
    n_cases: int = Field(default=32, ge=4)


class CounterexampleSpec(Strict):
    enabled: bool = True
    budget_forwards: int = Field(default=600, ge=10)
    seeds: list[int] = [0, 1, 2]
    strategies: list[Literal["random", "coverage", "guided"]] = ["random", "coverage", "guided"]
    slice_cases: int = Field(default=24, ge=4)
    minimize: bool = True


class ChallengeSpec(Strict):
    numerical: NumericalSpec | None = None
    symmetry: SymmetrySpec | None = None
    counterexamples: CounterexampleSpec | None = None


class ExperimentSpec(Strict):
    experiment_id: str
    claim: str
    model: str
    model_revision: str
    device: str = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    seed: int = 0
    deterministic: bool = True
    task: TaskSpec = TaskSpec()
    baseline: BaselineSpec = BaselineSpec()
    challenges: ChallengeSpec = ChallengeSpec()


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def load_claim(path: str | Path) -> ClaimCard:
    return ClaimCard.model_validate(_load_yaml(Path(path)))


def load_experiment(path: str | Path) -> tuple[ExperimentSpec, ClaimCard]:
    """Load an experiment and its claim, and check they describe the same model."""
    path = Path(path)
    spec = ExperimentSpec.model_validate(_load_yaml(path))
    claim_path = Path(spec.claim)
    if not claim_path.is_absolute():
        claim_path = (path.parent / claim_path).resolve()
    claim = load_claim(claim_path)
    if claim.model != spec.model:
        raise ValueError(f"claim is about {claim.model} but experiment loads {spec.model}")
    if claim.model_revision != spec.model_revision:
        raise ValueError(
            f"claim pins revision {claim.model_revision} but experiment loads {spec.model_revision}"
        )
    return spec, claim


def dump_yaml(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False)
