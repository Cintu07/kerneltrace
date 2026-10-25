# KernelTrace — research starter v0.1

**Question:** When weights and prompts are fixed, do causal attribution measurements change across inference implementations? If so, are differences scientifically meaningful rather than measurement noise?

This repo is a **working experimental scaffold**, **not** a published result, an established safety vulnerability, or a full circuit discovery method.

## First run on an 8 GB RTX 5060

```powershell
# Create and activate a Python 3.11 environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install a CUDA-enabled PyTorch build using the instructions at https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python experiment.py --model openai-community/gpt2 --backends eager sdpa --max-cases 4
python -m pytest -q
```

On Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead. Hugging Face weights download on first run. `--revision main` is convenient for initial testing but must be replaced with a pinned commit hash for a publishable experiment.

## What it measures

- fixed model and weights, FP32, TF32 disabled, dropout disabled
- paired clean / corrupted prompts with matched token lengths
- clean and corrupted answer-minus-distractor logits
- causal restoration by replacing **one MLP block output** from the clean run into the corrupted run
- per-layer causal effect and normalized restoration
- top-3 overlap and max absolute effect difference across eager vs SDPA backends
- model, backend, machine, library, commit metadata

## What it does NOT measure

- full neuron/edge/attention-head circuits (v0.1 uses layer-level MLP blocks)
- safety/refusal mechanisms
- numerical error ground truth or high-precision reference
- statistical confidence from multiple prompts, seeds, templates or architectures
- whether particular backend kernels were actually selected: verify with a profiler
- whether output text is exactly the same (next-token margin is recorded as a weak starting proxy)

## Data quality gate

**Do not interpret attribution differences without a meaningful clean/corrupt contrast.** Check both `clean_margin` and `corrupt_margin` and especially `contrast`. Initial prompts are illustrative and may not elicit strong IOI behavior in GPT-2. Replace or scale the fixture if the contrast is weak. Some individual cases may fail this check.

## Research expansion checklist

1. Reproduce a known circuit on a standard IOI / Greater-Than benchmark.
2. Use prompts that pass independently defined behavioural contrasts.
3. Instrument the exact CUDA kernels executed via profiler.
4. Add replicate runs and backend-specific numerical controls.
5. Separate measurement error from true model computation changes.
6. Compare full logit distributions, rank correlations, and causal faithfulness on held-out prompts.
7. Add FP64 reference where supported, bootstrap intervals, across-model evaluation and multiple independent task templates.
8. Study whether results predict **failure** of an explanation under changed conditions.

## Research integrity

Negative results are welcome. Do not claim a circuit changed just because top-k rankings changed. Do not claim ICML novelty until a complete literature review is finished. Original results must be run and verified on real hardware.
