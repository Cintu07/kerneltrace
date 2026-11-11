"""intervention fidelity (fr-03). null patches come back null, and a toy oracle is recovered."""
import numpy as np
import pytest
import torch

from kerneltrace.interventions import patching as P

from .conftest import fake_cases, tiny_gpt2, tiny_neox


def test_identity_patch_is_exactly_null(gpt2, cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    be = P.block_effects(gpt2, batch, identity_controls=True)
    assert np.array_equal(be.identity, np.broadcast_to(be.corrupt, be.identity.shape))


def test_clean_into_clean_returns_clean(gpt2, cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    _, cache = P.run(gpt2, batch.clean, capture=[("mlp_out", 1)])
    logits, _ = P.run(gpt2, batch.clean, patches={("mlp_out", 1): P.Patch(cache[("mlp_out", 1)])})
    ref, _ = P.run(gpt2, batch.clean)
    assert torch.equal(logits, ref)


def test_toy_oracle_recovers_known_effect():
    """with attention and the later mlps zeroed, a layer-0 mlp patch has a closed form."""
    a = tiny_gpt2(seed=1)
    with torch.no_grad():
        for i, block in enumerate(a.blocks()):
            block.attn.c_proj.weight.zero_()
            block.attn.c_proj.bias.zero_()
            if i > 0:
                block.mlp.c_proj.weight.zero_()
                block.mlp.c_proj.bias.zero_()
    cases = fake_cases(4, 8, seed=2)
    batch = P.make_batches(cases, 8, "cpu")[0]
    patched = P.joint_patch_margins(a, batch, [0])

    m = a.model
    with torch.no_grad():
        pos = torch.arange(batch.clean.shape[1])
        emb_x = m.transformer.wte(batch.corrupt) + m.transformer.wpe(pos)
        emb_c = m.transformer.wte(batch.clean) + m.transformer.wpe(pos)
        mlp0_clean = a.mlp(0)(a.blocks()[0].ln_2(emb_c))
        h = emb_x + mlp0_clean
        logits = m.lm_head(m.transformer.ln_f(h))
        expected = P.margins(logits, batch.a_ids, batch.b_ids).numpy()
    np.testing.assert_allclose(patched, expected, atol=1e-5)


def test_patch_shape_mismatch_raises(gpt2, cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    with pytest.raises(RuntimeError, match="shape"):
        P.run(gpt2, batch.corrupt, patches={("mlp_out", 0): P.Patch(torch.zeros(1, 2, 3))})


def test_hooks_are_removed(gpt2, cases):
    batch = P.make_batches(cases, 8, "cpu")[0]
    P.block_effects(gpt2, batch)
    for layer in range(gpt2.n_layers):
        assert not gpt2.mlp(layer)._forward_hooks
        assert not gpt2.mlp_act(layer)._forward_hooks


def test_batches_never_mix_lengths():
    cases = fake_cases(5, 10) + fake_cases(3, 7, seed=5)
    for c in cases[5:]:
        object.__setattr__(c.case, "case_id", c.case.case_id + "b")
    batches = P.make_batches(cases, 4, "cpu")
    assert sum(len(b.cases) for b in batches) == 8
    for b in batches:
        assert len({c.length for c in b.cases}) == 1


def test_batched_equals_unbatched_up_to_tolerance(gpt2, cases):
    big = P.make_batches(cases, 8, "cpu")[0]
    be_big = P.block_effects(gpt2, big, identity_controls=False)
    singles = [P.block_effects(gpt2, b, identity_controls=False) for b in P.make_batches(cases, 1, "cpu")]
    np.testing.assert_allclose(be_big.patched, np.concatenate([s.patched for s in singles], axis=1), atol=1e-4)


def test_neuron_attribution_matches_exact_patch_for_small_change(gpt2, cases):
    """attribution patching is a first-order estimate. its sign should agree with exact patching
    on the channels it scores highest."""
    batch = P.make_batches(cases, 8, "cpu")[0]
    scores = P.neuron_attribution(gpt2, batch, 1).mean(0)
    top = torch.as_tensor(np.argsort(-np.abs(scores))[:4].copy())
    exact = P.neuron_patch_margins(gpt2, batch, 1, top)
    corrupt = P.margins(P.run(gpt2, batch.corrupt)[0], batch.a_ids, batch.b_ids).numpy()
    assert np.sign((exact - corrupt).mean()) == np.sign(scores[top.numpy()].sum())


@pytest.mark.parametrize("make", [tiny_gpt2, tiny_neox])
def test_fp32_logits_matches_the_normal_head_in_fp32(make, cases):
    a = make()
    batch = P.make_batches(cases, 8, "cpu")[0]
    site = ("mlp_out", 1)
    _, cache = P.run(a, batch.clean, capture=[site])
    ref, _ = P.run(a, batch.corrupt, patches={site: P.Patch(cache[site])})
    got, _ = P.run(a, batch.corrupt, patches={site: P.Patch(cache[site])}, fp32_logits=True)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_fp32_logits_in_bf16_lands_closer_to_fp32():
    """a bf16 body with an fp32 head should sit nearer the fp32 reference than plain bf16."""
    ref_model = tiny_gpt2(seed=3)
    low = tiny_gpt2(seed=3)
    low.model.to(torch.bfloat16)
    ids = torch.randint(4, 64, (4, 10), generator=torch.Generator().manual_seed(0))
    ref, _ = P.run(ref_model, ids)
    plain, _ = P.run(low, ids)
    fixed, _ = P.run(low, ids, fp32_logits=True)
    assert (fixed - ref).abs().max() <= (plain.float() - ref).abs().max()


@pytest.mark.parametrize("make", [tiny_gpt2, tiny_neox])
def test_upcast_attention_runs_attention_in_fp32_and_hands_back_bf16(make, cases):
    from kerneltrace.models.precision import upcast_attention

    a = make()
    a.model.to(torch.bfloat16)
    seen = []
    upcast_attention(a)
    a.attention(0).register_forward_hook(lambda _m, i, o: seen.append(next(p.dtype for p in _m.parameters())))
    batch = P.make_batches(cases, 8, "cpu")[0]
    logits, cache = P.run(a, batch.corrupt, capture=[("mlp_out", 0)])
    assert seen == [torch.float32]
    assert cache[("mlp_out", 0)].dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all()


def test_upcast_attention_is_a_no_op_in_fp32(cases):
    from kerneltrace.models.precision import upcast_attention

    a, b = tiny_neox(seed=4), tiny_neox(seed=4)
    upcast_attention(b)
    batch = P.make_batches(cases, 8, "cpu")[0]
    assert torch.equal(P.run(a, batch.corrupt)[0], P.run(b, batch.corrupt)[0])


@pytest.mark.parametrize("make", [tiny_gpt2, tiny_neox])
def test_dropping_the_invariant_key_bias_leaves_logits_alone(make, cases):
    from kerneltrace.models.precision import drop_invariant_key_bias

    a = make(seed=5)
    with torch.no_grad():
        for layer in range(a.n_layers):
            a.invariant_key_bias(layer).normal_(0, 30.0)  # big, like pythia's
    batch = P.make_batches(cases, 8, "cpu")[0]
    before, _ = P.run(a, batch.corrupt)
    removed = drop_invariant_key_bias(a)
    after, _ = P.run(a, batch.corrupt)
    assert removed > 50
    for layer in range(a.n_layers):
        assert not a.invariant_key_bias(layer).any()
    torch.testing.assert_close(after, before, atol=1e-3, rtol=1e-4)


def test_the_rest_of_the_neox_key_bias_is_not_invariant(cases):
    """the rotary part of the key bias does change attention, so it has to stay."""
    a = tiny_neox(seed=6)
    attn = a.attention(0)
    heads, size = a.config.num_attention_heads, a.config.hidden_size // a.config.num_attention_heads
    batch = P.make_batches(cases, 8, "cpu")[0]
    before, _ = P.run(a, batch.corrupt)
    with torch.no_grad():
        attn.query_key_value.bias.view(heads, 3, size)[:, 1, :attn.rotary_ndims].normal_(0, 30.0)
    after, _ = P.run(a, batch.corrupt)
    assert (after - before).abs().max() > 1e-2
