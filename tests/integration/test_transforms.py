"""symmetry validation (fr-05) and dispatch probing (fr-04)."""
import copy

import numpy as np
import pytest
import torch

from kerneltrace.core.spec import ConditionSpec
from kerneltrace.interventions import patching as P
from kerneltrace.models.adapter import UnsupportedModel, wrap
from kerneltrace.transforms import backend as B
from kerneltrace.transforms import symmetry as SYM

from .conftest import tiny_gpt2, tiny_neox


@pytest.mark.parametrize("make", [tiny_gpt2, tiny_neox], ids=["gpt2", "gpt_neox"])
def test_permutation_preserves_function(make, cases):
    a = make()
    perms = SYM.random_permutations(a.n_layers, a.d_mlp, seed=3)
    pm = SYM.permuted_copy(a, perms)
    batch = P.make_batches(cases, 8, "cpu")[0]
    la, _ = P.run(a, batch.clean)
    lb, _ = P.run(pm.adapter, batch.clean)
    parity = SYM.logit_parity(la, lb, atol=1e-4)
    assert parity.passed, parity
    # the weights really changed, so this pass means something.
    assert not torch.equal(a.mlp(0).state_dict()[next(iter(a.mlp(0).state_dict()))],
                           pm.adapter.mlp(0).state_dict()[next(iter(pm.adapter.mlp(0).state_dict()))])


@pytest.mark.parametrize("make", [tiny_gpt2, tiny_neox], ids=["gpt2", "gpt_neox"])
def test_hidden_channels_move_as_mapped(make, cases):
    a = make()
    perms = SYM.random_permutations(a.n_layers, a.d_mlp, seed=4)
    pm = SYM.permuted_copy(a, perms)
    batch = P.make_batches(cases, 8, "cpu")[0]
    _, ca = P.run(a, batch.clean, capture=[("mlp_hidden", 1)])
    _, cb = P.run(pm.adapter, batch.clean, capture=[("mlp_hidden", 1)])
    torch.testing.assert_close(cb[("mlp_hidden", 1)], ca[("mlp_hidden", 1)][..., perms[1]], atol=1e-5, rtol=0)


def test_mapped_attributions_agree_and_naive_ids_are_flagged(gpt2, cases):
    perms = SYM.random_permutations(gpt2.n_layers, gpt2.d_mlp, seed=5)
    pm = SYM.permuted_copy(gpt2, perms)
    batch = P.make_batches(cases, 8, "cpu")[0]
    so = P.neuron_attribution(gpt2, batch, 1).mean(0)
    st = P.neuron_attribution(pm.adapter, batch, 1).mean(0)
    cmp = SYM.mapped_comparison(so, st, perms[1])
    assert cmp["mapped_spearman"] > 0.999
    assert cmp["mapped_max_abs_diff"] < 1e-5
    assert cmp["naive_spearman"] < 0.5


def test_maps_round_trip(gpt2):
    pm = SYM.permuted_copy(gpt2, SYM.random_permutations(gpt2.n_layers, gpt2.d_mlp, seed=6))
    idx = torch.arange(gpt2.d_mlp)
    assert torch.equal(pm.to_original(1, pm.to_transformed(1, idx)), idx)


def test_non_permutation_rejected(gpt2):
    bad = {0: torch.zeros(gpt2.d_mlp, dtype=torch.long)}
    with pytest.raises(ValueError):
        SYM.permuted_copy(gpt2, bad)


def test_parity_gate_fails_on_a_broken_transform(gpt2, cases):
    broken = copy.deepcopy(gpt2.model)
    with torch.no_grad():
        broken.transformer.h[0].mlp.c_fc.weight[:, :3] *= -1  # no matching inverse applied
    batch = P.make_batches(cases, 8, "cpu")[0]
    la, _ = P.run(gpt2, batch.clean)
    lb, _ = P.run(wrap(broken), batch.clean)
    assert not SYM.logit_parity(la, lb, atol=1e-4).passed


def test_unsupported_architecture_refused():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=4)
    with pytest.raises(UnsupportedModel):
        wrap(LlamaForCausalLM(cfg))


def test_probe_sees_eager_and_sdpa(cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    eager = B.probe(tiny_gpt2("eager"), batch.clean, ConditionSpec(name="e"))
    assert eager["verified"], eager["problems"]
    assert eager["observed_attention"] == "eager_matmul_softmax"
    sdpa = B.probe(tiny_gpt2("sdpa"), batch.clean, ConditionSpec(name="s", attn_implementation="sdpa"))
    assert sdpa["verified"], sdpa["problems"]
    assert sdpa["observed_attention"] not in ("eager_matmul_softmax", "unknown")


def test_probe_rejects_mislabelled_condition(cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    res = B.probe(tiny_gpt2("eager"), batch.clean, ConditionSpec(name="x", attn_implementation="sdpa"))
    assert not res["verified"]


def test_forced_math_kernel_is_observed(cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    res = B.probe(tiny_gpt2("sdpa"), batch.clean,
                  ConditionSpec(name="m", attn_implementation="sdpa", sdpa_kernel="math"))
    assert res["verified"], res
    assert "math" in res["observed_attention"]


def test_factor_labelling():
    ref = ConditionSpec(name="r")
    assert B.factors(ref, ConditionSpec(name="a")) == ["repeat"]
    assert B.factors(ref, ConditionSpec(name="a", attn_implementation="sdpa")) == ["attention_backend"]
    assert B.factors(ref, ConditionSpec(name="a", dtype="bfloat16", tf32=True)) == ["dtype", "tf32"]
    assert B.factors(ref, ConditionSpec(name="a", batch_size=1)) == ["batching"]


def test_execution_context_restores_tf32():
    before = torch.backends.cuda.matmul.allow_tf32
    with B.execution(ConditionSpec(name="t", tf32=not before)):
        assert torch.backends.cuda.matmul.allow_tf32 == (not before)
    assert torch.backends.cuda.matmul.allow_tf32 == before


def test_sdpa_and_eager_agree_on_tiny_model(cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    e = tiny_gpt2("eager", seed=7)
    s = tiny_gpt2("sdpa", seed=7)
    la, _ = P.run(e, batch.clean)
    lb, _ = P.run(s, batch.clean)
    assert np.allclose(la.numpy(), lb.numpy(), atol=1e-5)


def test_parity_relative_tolerance_scales_with_logit_size():
    a = torch.tensor([[800.0, -5.0]])
    b = torch.tensor([[800.0 + 2e-3, -5.0]])  # ~30 fp32 ulps at 800: summation-order noise
    assert not SYM.logit_parity(a, b, atol=1e-3).passed
    assert SYM.logit_parity(a, b, atol=1e-3, rtol=1e-5).passed
    worse = torch.tensor([[800.0, -5.0 + 1e-2]])  # a small logit off by far more than its scale allows
    assert not SYM.logit_parity(a, worse, atol=1e-3, rtol=1e-5).passed
