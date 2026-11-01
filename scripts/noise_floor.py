"""is a permutation's parity gap bigger than the model's own fp32 noise floor?

compares permuted-vs-original logits with the same model run at different batch sizes.
usage: python scripts/noise_floor.py   (needs cuda and the pinned weights)
"""
import sys, torch
from kerneltrace.core.manifest import configure_determinism
from kerneltrace.models.adapter import load, load_tokenizer
from kerneltrace.transforms import symmetry as SYM
from kerneltrace.tasks import ioi
configure_determinism(0, True)
for model, rev, cfg_seed in [("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46", 1),
                             ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", 1)]:
    a = load(model, rev, "eager", "float32", "cuda"); tok = load_tokenizer(model, rev)
    cases = [c for c in ioi.generate("discovery", 64, 0)]
    by_len = {}
    for c in cases:
        ids = tok.encode(c.clean)
        by_len.setdefault(len(ids), []).append(ids)
    L, group = max(by_len.items(), key=lambda kv: len(kv[1]))
    x = torch.tensor(group[:16], device="cuda")
    pm = SYM.permuted_copy(a, SYM.random_permutations(a.n_layers, a.d_mlp, cfg_seed))
    with torch.no_grad():
        base = a.model(input_ids=x).logits
        perm = pm.adapter.model(input_ids=x).logits
        single = torch.cat([a.model(input_ids=x[i:i+1]).logits for i in range(len(x))])
        half = torch.cat([a.model(input_ids=x[:8]).logits, a.model(input_ids=x[8:]).logits])
    def d(u, v):
        p = SYM.logit_parity(u, v, 1e-3)
        diff = (u - v).abs(); i = diff.argmax()
        return f"max {p.max_abs_logit_diff:.2e} (at |logit| {u.flatten()[i].abs().item():.0f}), mean {p.mean_abs_logit_diff:.1e}, maxKL {p.max_kl:.1e}, argmax {p.argmax_agreement:.3f}"
    print(f"== {model}  batch of {len(x)} prompts, length {L}, max |logit| {base.abs().max().item():.0f}")
    print("  permuted vs original      :", d(base, perm))
    print("  batch 1 vs batch 16 (same model, nothing permuted):", d(base, single))
    print("  batch 8 vs batch 16 (same model, nothing permuted):", d(base, half))
    del a, pm; torch.cuda.empty_cache()
