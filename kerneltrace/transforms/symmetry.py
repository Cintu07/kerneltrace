"""function-preserving mlp channel permutations.

permute an mlp's hidden channels, apply the inverse permutation to the down
projection, and the network computes the same function in exact arithmetic,
because the nonlinearity is elementwise. in floating point the down-projection
sums run in a different order, so output parity gets measured, and nothing is
compared until parity passes.

the permutation comes back as an explicit component map. a claim about
"neuron 1234" only means something through that map.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from ..models.adapter import ModelAdapter, wrap


@dataclass
class PermutedModel:
    adapter: ModelAdapter
    perms: dict[int, torch.Tensor]  # layer -> perm; transformed j == original perm[j]

    def to_original(self, layer: int, transformed: torch.Tensor) -> torch.Tensor:
        return self.perms[layer].to(transformed.device)[transformed]

    def to_transformed(self, layer: int, original: torch.Tensor) -> torch.Tensor:
        inverse = torch.argsort(self.perms[layer])
        return inverse.to(original.device)[original]


def random_permutations(n_layers: int, d_mlp: int, seed: int) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {layer: torch.randperm(d_mlp, generator=g) for layer in range(n_layers)}


def permuted_copy(adapter: ModelAdapter, perms: dict[int, torch.Tensor]) -> PermutedModel:
    clone = wrap(copy.deepcopy(adapter.model))
    for layer, perm in perms.items():
        if sorted(perm.tolist()) != list(range(adapter.d_mlp)):
            raise ValueError(f"layer {layer}: not a permutation of {adapter.d_mlp} channels")
        clone.permute_mlp(layer, perm)
    return PermutedModel(clone, perms)


@dataclass
class Parity:
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    max_kl: float
    argmax_agreement: float
    atol: float
    rtol: float = 0.0
    max_excess_over_tolerance: float = 0.0  # max(|a - b| - (atol + rtol * |a|)); <= 0 passes

    @property
    def passed(self) -> bool:
        return self.max_excess_over_tolerance <= 0 and self.argmax_agreement == 1.0


def logit_parity(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float = 0.0) -> Parity:
    """compare two [B, V] (or [B, T, V]) logit tensors over the full vocabulary.

    the tolerance is atol + rtol * |a| per logit. fp32 summation-order noise grows with
    the size of the logit, so a purely absolute tolerance tuned on one model can fail
    on a model with bigger logits even when the transform is exact.
    """
    a = a.double().reshape(-1, a.shape[-1])
    b = b.double().reshape(-1, b.shape[-1])
    diff = (a - b).abs()
    excess = float((diff - (atol + rtol * a.abs())).max())
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    return Parity(
        max_abs_logit_diff=float(diff.max()),
        mean_abs_logit_diff=float(diff.mean()),
        max_kl=float(kl.max()),
        argmax_agreement=float((a.argmax(-1) == b.argmax(-1)).double().mean()),
        atol=atol,
        rtol=rtol,
        max_excess_over_tolerance=excess,
    )


def mapped_comparison(original_scores: np.ndarray, transformed_scores: np.ndarray,
                      perm: torch.Tensor) -> dict:
    """compare per-channel scores with and without the component map.

    original_scores[i] is channel i in the original model. transformed_scores[j] is
    channel j in the permuted model, which is original perm[j].
    """
    from scipy.stats import spearmanr

    p = perm.cpu().numpy()
    mapped = original_scores[p]
    naive = original_scores
    return {
        "mapped_spearman": float(spearmanr(mapped, transformed_scores).statistic),
        "mapped_max_abs_diff": float(np.abs(mapped - transformed_scores).max()),
        "naive_spearman": float(spearmanr(naive, transformed_scores).statistic),
        "naive_max_abs_diff": float(np.abs(naive - transformed_scores).max()),
    }
