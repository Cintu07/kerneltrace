#!/usr/bin/env bash
# Runs every pre-registered experiment in sequence with the current code.
cd "$(dirname "$0")/.."
for spec in experiments/gpt2_ioi_mlp0.yaml experiments/gpt2_ioi_late_mlp.yaml experiments/pythia160m_ioi_mlp0.yaml; do
  name=$(basename "$spec" .yaml)
  HF_HUB_DISABLE_SYMLINKS_WARNING=1 ./.venv/Scripts/kerneltrace run "$spec" > "runs_${name}.log" 2>&1
  echo "$name EXIT=$?"
done
