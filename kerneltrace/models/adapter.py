"""model adapters.

an adapter names the hook points an experiment is allowed to touch, and knows
how to apply weight-space transforms for its architecture. a model without an
adapter gets refused outright.
"""
from __future__ import annotations

import torch
from torch import nn

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class UnsupportedModel(TypeError):
    pass


class ModelAdapter:
    family = "base"

    def __init__(self, model: nn.Module):
        self.model = model.eval()

    @property
    def config(self):
        return self.model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def attn_implementation(self) -> str:
        return getattr(self.model.config, "_attn_implementation", "unknown")

    @property
    def n_layers(self) -> int:
        return len(self.blocks())

    def blocks(self) -> nn.ModuleList:
        raise NotImplementedError

    def mlp(self, layer: int) -> nn.Module:
        """module whose output is the mlp's contribution to the residual stream."""
        raise NotImplementedError

    def mlp_act(self, layer: int) -> nn.Module:
        """module whose output is the mlp hidden state after the nonlinearity."""
        raise NotImplementedError

    @property
    def d_mlp(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def permute_mlp(self, layer: int, perm: torch.Tensor) -> None:
        """reorder the hidden channels of one mlp. hidden j afterwards is hidden perm[j] before."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "family": self.family,
            "model_type": self.config.model_type,
            "n_layers": self.n_layers,
            "d_mlp": self.d_mlp,
            "attn_implementation": self.attn_implementation,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
            "commit": getattr(self.config, "_commit_hash", None),
            "name_or_path": getattr(self.config, "_name_or_path", None),
        }


class GPT2Adapter(ModelAdapter):
    """gpt-2. its Conv1D layers store weights as [in, out]."""

    family = "gpt2"

    def blocks(self):
        return self.model.transformer.h

    def mlp(self, layer):
        return self.blocks()[layer].mlp

    def mlp_act(self, layer):
        return self.blocks()[layer].mlp.act

    @property
    def d_mlp(self):
        return self.blocks()[0].mlp.c_fc.weight.shape[1]

    @torch.no_grad()
    def permute_mlp(self, layer, perm):
        mlp = self.mlp(layer)
        perm = perm.to(mlp.c_fc.weight.device)
        mlp.c_fc.weight.copy_(mlp.c_fc.weight[:, perm])
        mlp.c_fc.bias.copy_(mlp.c_fc.bias[perm])
        mlp.c_proj.weight.copy_(mlp.c_proj.weight[perm, :])


class GPTNeoXAdapter(ModelAdapter):
    """gpt-neox and pythia. nn.Linear layers store weights as [out, in]."""

    family = "gpt_neox"

    def blocks(self):
        return self.model.gpt_neox.layers

    def mlp(self, layer):
        return self.blocks()[layer].mlp

    def mlp_act(self, layer):
        return self.blocks()[layer].mlp.act

    @property
    def d_mlp(self):
        return self.blocks()[0].mlp.dense_h_to_4h.weight.shape[0]

    @torch.no_grad()
    def permute_mlp(self, layer, perm):
        mlp = self.mlp(layer)
        perm = perm.to(mlp.dense_h_to_4h.weight.device)
        mlp.dense_h_to_4h.weight.copy_(mlp.dense_h_to_4h.weight[perm, :])
        mlp.dense_h_to_4h.bias.copy_(mlp.dense_h_to_4h.bias[perm])
        mlp.dense_4h_to_h.weight.copy_(mlp.dense_4h_to_h.weight[:, perm])


ADAPTERS = {"gpt2": GPT2Adapter, "gpt_neox": GPTNeoXAdapter}


def wrap(model: nn.Module) -> ModelAdapter:
    model_type = model.config.model_type
    if model_type not in ADAPTERS:
        raise UnsupportedModel(
            f"no adapter for model_type={model_type!r}; supported: {sorted(ADAPTERS)}"
        )
    adapter = ADAPTERS[model_type](model)
    _check_hooks(adapter)
    return adapter


def _check_hooks(adapter: ModelAdapter) -> None:
    """every hook point the experiments use has to exist on every layer."""
    for layer in range(adapter.n_layers):
        for name, get in (("mlp", adapter.mlp), ("mlp_act", adapter.mlp_act)):
            mod = get(layer)
            if not isinstance(mod, nn.Module):
                raise UnsupportedModel(f"layer {layer} {name} is not a hookable module")


def load(model_id: str, revision: str, attn_implementation: str = "eager",
         dtype: str = "float32", device: str = "cpu") -> ModelAdapter:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, attn_implementation=attn_implementation,
        dtype=DTYPES[dtype],
    ).to(device)
    adapter = wrap(model)
    if adapter.attn_implementation != attn_implementation:
        raise RuntimeError(
            f"requested attn_implementation={attn_implementation} but config reports "
            f"{adapter.attn_implementation}"
        )
    for p in adapter.model.parameters():
        p.requires_grad_(False)
    return adapter


def load_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision)
