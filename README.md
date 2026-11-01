# KernelTrace

**Find the explanation. Try to break it. Keep the evidence.**

![kerneltrace method](docs/figures/kerneltrace_method.png)

KernelTrace is a local-first lab for testing mechanistic interpretability claims instead of
just producing them. You write the claim down first, as a claim card with explicit predictions
and the measurements that would count against it. KernelTrace then measures it on held-out
data and attacks it with challenges the claim did not anticipate:

- **numerical execution**: does the explanation survive a different attention kernel, batch
  size, TF32, or a rerun? The tool checks with the profiler which kernel actually ran.
- **representation symmetry**: does it survive a function-preserving permutation of every MLP's
  hidden channels, once component identities are mapped correctly?
- **counterexample search**: is there a valid input, one the model really solves, where the
  prediction fails? If so, it shrinks that input to the smallest change that still breaks the
  claim and tests whether the failure generalises to a sub-scope.

Every run ends in a status of `supported_in_scope`, `challenged`, `falsified_in_scope` or
`inconclusive`, and an evidence bundle another researcher can inspect and replay. A failed
prerequisite (too few valid cases, an intervention control that is not null, an unpinned model,
a kernel that did not run) never turns into a pass.

The design document is [docs/PRD.md](docs/PRD.md). Current state and next steps are in
[docs/PROGRESS.md](docs/PROGRESS.md).

## First results

Three claims were pre-registered (committed before any measurement, see the git history of
`experiments/claims/`) and tested on an RTX 5060 Laptop GPU. Full write-up with every number:
[docs/FINDINGS.md](docs/FINDINGS.md). Bundles: [evidence/](evidence/).

| claim | model | status | why |
|---|---|---|---|
| Restoring MLP 0 restores at least half the IOI logit contrast | GPT-2 small | `supported_in_scope` | held-out restoration 0.84 [0.82, 0.86]; survived every listed challenge |
| Late MLPs 9-11 jointly carry the IOI decision | GPT-2 small | `falsified_in_scope` | held-out restoration 0.084 [0.076, 0.092] against 0.5 |
| Same MLP 0 claim on another architecture | Pythia-160m | `inconclusive` | effect replicates (0.997), but the permutation parity gate failed, so a listed invariance went untested |

At fp32, switching attention kernels (eager, SDPA math, SDPA memory-efficient) or batch
size moved GPT-2's causal measurements by at most 2e-4 logits and changed no decision. bf16
changed which blocks get selected. The Pythia parity failure turned out to be the same size as
Pythia's own batch-to-batch fp32 noise, which shows the parity gate needs to be calibrated
against a measured noise floor. That is the next change.

![verdicts](docs/figures/fig2_verdicts.png)

These are block-level statements about one task family. They are measurements of where
restoring a clean activation matters, not a description of the algorithm.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the build for your GPU
pip install -e ".[dev]"

pytest -q                                            # 62 tests, CPU only, no downloads
kerneltrace validate experiments/claims/*.yaml experiments/*.yaml
kerneltrace run experiments/gpt2_ioi_mlp0.yaml       # about 6 minutes on an 8 GB laptop GPU
kerneltrace replay evidence/<bundle>                 # rerun a published bundle's reference measurement
```

Open `runs/<run_id>/report.html` for the report.

## A claim card

```yaml
claim_id: gpt2_ioi_mlp0_extended_embedding_v1
hypothesis: In GPT-2 small on IOI with an S2-flip corruption, MLP block 0 acts as an
  extended token embedding ...
model: openai-community/gpt2
model_revision: 607a30d783dfa663caf39e06633721c8d4cfcd7e
predictions:
  - id: mean_restoration
    kind: restoration_at_least
    components: [0]
    threshold: 0.5
    quantifier: mean          # tested with a template-cluster bootstrap CI
    split: holdout
  - id: per_case_restoration
    kind: restoration_at_least
    components: [0]
    threshold: 0.25
    quantifier: per_case      # the prediction a single input can contradict
    max_violation_rate: 0.05
invariances:
  - kind: attention_backend
  - kind: batching
  - kind: tf32
  - kind: mlp_channel_permutation
  - kind: counterfactual_inputs
falsification:
  - held-out mean restoration CI entirely below 0.5
  - ...
```

An invariance listed in the claim that no valid challenge tested blocks a `supported` status.
Claim thresholds are fixed before running; a changed claim gets a new `claim_id`.

## What a run produces

```
runs/<run_id>/
  manifest.json          model commit, torch/CUDA/driver, GPU, TF32 and determinism switches, git state
  claim.yaml             the claim with its resulting status
  experiment.yaml        the spec, so the bundle replays on its own
  preflight.json         every generated case, why rejected cases were rejected, the gates
  raw_results.parquet    per case, per layer, per condition (also .csv)
  interventions.jsonl    what was patched, where, from which run into which
  metrics.json           predictions, intervals, controls, every challenge
  counterexamples.jsonl  each counterexample with its replay result
  profiler/              observed attention kernels per condition
  figures/               SVG
  report.md, report.html
  files.json             sha256 of every file above
```

## How the status is decided

In order: a failed gate gives `inconclusive`; a prediction whose interval lies on the wrong side
of its threshold gives `falsified_in_scope`; one whose interval straddles it gives
`inconclusive`; an invariance the claim lists but no valid challenge tested gives
`inconclusive`; a valid challenge that contradicts the claim gives `challenged`; otherwise
`supported_in_scope`. The rules live in [kerneltrace/core/verdict.py](kerneltrace/core/verdict.py).

A numerical condition counts as contradicting the claim only when it changes a decision: a
prediction outcome, or the blocks selected on the discovery split (unless the change crosses
a near tie). Numeric drift that changes no decision is reported but does not challenge the
claim. Changing the dtype is a different measurement from an fp32 claim, so it is reported as
out of scope.

## Scope and limits of v0.2

- Models: GPT-2 and GPT-NeoX/Pythia adapters. Other architectures are refused, not guessed at.
- Interventions: MLP block outputs (single and joint) and MLP hidden channels. No attention
  heads yet.
- One task family, IOI with an S2-flip corruption, 16 templates split 8/8.
- Windows PyTorch wheels have no flash attention, so that condition shows as not run there.
- Nothing here says anything about real-world model safety.

## Layout

See [AGENTS.md](AGENTS.md). The original v0.1 single-file scaffold is kept in
[legacy/](legacy/) for provenance.

## License

MIT
