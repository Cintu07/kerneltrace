"""activation capture and patching.

sites are ("mlp_out", layer) for a block's mlp output and ("mlp_hidden", layer)
for the hidden channels after the nonlinearity. a patch swaps a site's output
for a stored tensor, optionally on just a subset of hidden channels. every patch
checks shapes and counts how often it fired, so a patch that silently did
nothing raises an error.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.adapter import ModelAdapter
from ..tasks.preflight import TokenizedCase

Site = tuple[str, int]


def site_module(adapter: ModelAdapter, site: Site):
    kind, layer = site
    if kind == "mlp_out":
        return adapter.mlp(layer)
    if kind == "mlp_hidden":
        return adapter.mlp_act(layer)
    raise ValueError(f"unknown site kind {kind!r}")


@dataclass
class Patch:
    value: torch.Tensor
    channels: torch.Tensor | None = None  # last-dim indices to replace, None means all of them
    fired: int = field(default=0, init=False)


@contextmanager
def _hooks(pairs):
    handles = [mod.register_forward_hook(fn) for mod, fn in pairs]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _capture_hook(store: dict, site: Site):
    def hook(_mod, _inp, out):
        store[site] = out.detach().clone()
    return hook


def _patch_hook(site: Site, patch: Patch):
    def hook(_mod, _inp, out):
        if out.shape != patch.value.shape:
            raise RuntimeError(f"patch shape {tuple(patch.value.shape)} != activation "
                               f"{tuple(out.shape)} at {site}")
        patch.fired += 1
        if patch.channels is None:
            return patch.value.to(out.dtype)
        new = out.clone()
        new[..., patch.channels] = patch.value[..., patch.channels].to(out.dtype)
        return new
    return hook


@torch.no_grad()
def run(adapter: ModelAdapter, input_ids: torch.Tensor, capture: list[Site] = (),
        patches: dict[Site, Patch] | None = None) -> tuple[torch.Tensor, dict[Site, torch.Tensor]]:
    """one forward pass, capturing and/or patching sites. returns logits and captures."""
    patches = patches or {}
    store: dict[Site, torch.Tensor] = {}
    pairs = [(site_module(adapter, s), _patch_hook(s, p)) for s, p in patches.items()]
    # capture hooks go after patch hooks so they see the patched value.
    pairs += [(site_module(adapter, s), _capture_hook(store, s)) for s in capture]
    with _hooks(pairs):
        logits = adapter.model(input_ids=input_ids).logits
    for s, p in patches.items():
        if p.fired != 1:
            raise RuntimeError(f"patch at {s} fired {p.fired} times, expected 1")
    return logits, store


def margins(logits: torch.Tensor, a_ids: torch.Tensor, b_ids: torch.Tensor) -> torch.Tensor:
    """logit(answer) - logit(distractor) at the last position, in float64."""
    last = logits[:, -1, :].double()
    rows = torch.arange(last.shape[0], device=last.device)
    return last[rows, a_ids] - last[rows, b_ids]


@dataclass
class Batch:
    cases: list[TokenizedCase]
    clean: torch.Tensor
    corrupt: torch.Tensor
    a_ids: torch.Tensor
    b_ids: torch.Tensor

    @property
    def case_ids(self) -> list[str]:
        return [c.case.case_id for c in self.cases]


def make_batches(cases: list[TokenizedCase], batch_size: int, device) -> list[Batch]:
    """group equal-length cases, so there is no padding and no padding numerics either."""
    by_len: dict[int, list[TokenizedCase]] = {}
    for c in cases:
        by_len.setdefault(c.length, []).append(c)
    batches = []
    for length in sorted(by_len):
        group = by_len[length]
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            batches.append(Batch(
                cases=chunk,
                clean=torch.tensor([c.clean_ids for c in chunk], device=device),
                corrupt=torch.tensor([c.corrupt_ids for c in chunk], device=device),
                a_ids=torch.tensor([c.answer_id for c in chunk], device=device),
                b_ids=torch.tensor([c.distractor_id for c in chunk], device=device),
            ))
    return batches


def mlp_sites(adapter: ModelAdapter) -> list[Site]:
    return [("mlp_out", layer) for layer in range(adapter.n_layers)]


@dataclass
class BlockEffects:
    """per-case, per-layer results of single-block mlp patching on one batch."""

    case_ids: list[str]
    clean: np.ndarray            # [B]
    corrupt: np.ndarray          # [B]
    patched: np.ndarray          # [L, B] corrupt run with clean MLP output at layer l
    identity: np.ndarray | None  # [L, B] corrupt run with its own MLP output patched in
    clean_last_logits: torch.Tensor  # [B, V] for output parity checks


def block_effects(adapter: ModelAdapter, batch: Batch, identity_controls: bool = True) -> BlockEffects:
    sites = mlp_sites(adapter)
    clean_logits, clean_cache = run(adapter, batch.clean, capture=sites)
    corrupt_logits, corrupt_cache = run(adapter, batch.corrupt, capture=sites)
    m_clean = margins(clean_logits, batch.a_ids, batch.b_ids)
    m_corrupt = margins(corrupt_logits, batch.a_ids, batch.b_ids)
    patched, identity = [], []
    for site in sites:
        logits, _ = run(adapter, batch.corrupt, patches={site: Patch(clean_cache[site])})
        patched.append(margins(logits, batch.a_ids, batch.b_ids))
        if identity_controls:
            logits, _ = run(adapter, batch.corrupt, patches={site: Patch(corrupt_cache[site])})
            identity.append(margins(logits, batch.a_ids, batch.b_ids))
    return BlockEffects(
        case_ids=batch.case_ids,
        clean=m_clean.cpu().numpy(),
        corrupt=m_corrupt.cpu().numpy(),
        patched=torch.stack(patched).cpu().numpy(),
        identity=torch.stack(identity).cpu().numpy() if identity_controls else None,
        clean_last_logits=clean_logits[:, -1, :].float().cpu(),
    )


def joint_patch_margins(adapter: ModelAdapter, batch: Batch, layers: list[int],
                        clean_cache: dict | None = None) -> np.ndarray:
    """margins on the corrupt run with clean mlp outputs restored at all of `layers` at once."""
    sites = [("mlp_out", layer) for layer in layers]
    if clean_cache is None:
        _, clean_cache = run(adapter, batch.clean, capture=sites)
    logits, _ = run(adapter, batch.corrupt, patches={s: Patch(clean_cache[s]) for s in sites})
    return margins(logits, batch.a_ids, batch.b_ids).cpu().numpy()


def neuron_attribution(adapter: ModelAdapter, batch: Batch, layer: int) -> np.ndarray:
    """attribution patching, a first-order guess at each hidden channel's patch effect.

    score[b, n] = sum_t (h_clean - h_corrupt)[b, t, n] * d margin / d h[b, t, n],
    with the gradient taken on the corrupt run. returns [B, d_mlp].
    """
    site = ("mlp_hidden", layer)
    _, clean_cache = run(adapter, batch.clean, capture=[site])
    held = {}

    def hook(_mod, _inp, out):
        leaf = out.detach().requires_grad_(True)
        held["h"] = leaf
        return leaf

    with torch.enable_grad(), _hooks([(site_module(adapter, site), hook)]):
        logits = adapter.model(input_ids=batch.corrupt).logits
        m = margins(logits, batch.a_ids, batch.b_ids)
        m.sum().backward()
    h = held["h"]
    score = ((clean_cache[site] - h.detach()) * h.grad).sum(dim=1)
    return score.double().cpu().numpy()


def neuron_patch_margins(adapter: ModelAdapter, batch: Batch, layer: int,
                         channels: torch.Tensor) -> np.ndarray:
    """exact patch of a subset of hidden channels, clean into corrupt."""
    site = ("mlp_hidden", layer)
    _, clean_cache = run(adapter, batch.clean, capture=[site])
    logits, _ = run(adapter, batch.corrupt,
                    patches={site: Patch(clean_cache[site], channels=channels.to(adapter.device))})
    return margins(logits, batch.a_ids, batch.b_ids).cpu().numpy()
