# kerneltrace

![how kerneltrace works](docs/figures/x_method.png)

most interpretability results come from one experiment. you patch an activation, a heatmap lights up, and the story gets written down as "this component does x". nobody goes back and tries to break it.

kerneltrace is the thing that tries. you write a claim down first, with numbers, and commit it before you measure anything. then it gets measured on prompts it has never seen and attacked three ways. it reruns the model on other attention kernels, batch sizes, tf32 and bf16, and the profiler checks which kernel actually ran. it shuffles every mlp neuron in a way that keeps the math identical. and it searches thousands of prompts for one that breaks the claim, then shrinks that prompt down to the smallest change that still breaks it.

a claim comes out supported, falsified or inconclusive. if any check fails along the way, it lands on inconclusive. everything ends up in an evidence bundle with a hash for every file, so you can replay it.

## what came out

three claims, gpt-2 small and pythia-160m, the indirect object task, one laptop gpu.

![results](docs/figures/x_verdicts.png)

the gpt-2 claim that mlp block 0 restores at least half the answer held up. 0.84 on held-out prompts, and 13,500 forward passes of counterexample search found nothing. the claim that the last three mlp blocks carry the decision got 0.084 against a promised 0.5, so it's falsified. on pythia the mlp 0 effect replicates at 0.997, but the neuron shuffle drifted past its parity check, so it stays inconclusive. that drift turned out to be the same size as running the untouched model at a different batch size. the check needs a measured noise floor. a fixed number can't tell those apart, and that's the next fix.

the causal trace shows where the answer lives. restore one activation at one word and one layer at a time, and it sits on the repeated name for seven layers, then moves to the last word at layers 7 and 8. mlp 0 matters on that one word and nowhere else.

![causal trace](docs/figures/x_trace.png)

at fp32, swapping attention kernels or batch size moved the measurements by 2e-4 logits at most. bf16 changed which blocks rank top three, and 31% of the trace cells moved by 0.05 or more from rounding alone.

## run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the build for your gpu
pip install -e ".[dev]"
pytest -q                                          # 62 tests, cpu only, no downloads
kerneltrace run experiments/gpt2_ioi_mlp0.yaml     # about 6 minutes on an 8 gb laptop gpu
kerneltrace replay evidence/<bundle>               # rerun a bundle and compare it to what was stored
python scripts/causal_trace.py 24                  # the traces behind the picture above
```

a run writes a folder under `runs/`. inside is the manifest (model commit, library and driver versions, gpu, tf32 and determinism switches, git state), the claim with its final status, every generated prompt with the reason any were rejected, raw per-case results, the kernels the profiler saw, and a report.

## what's in here

`kerneltrace/` is the library: claim cards, the ioi task, gpt-2 and pythia adapters, patching, the three attacks, the stats and the verdict rules. `experiments/` has the claim cards and the specs that test them. `evidence/` holds the bundles from the runs above. `scripts/` draws the figures and computes the causal traces.

the status rules live in `kerneltrace/core/verdict.py`. short version: a failed gate means inconclusive. an interval on the wrong side of its threshold means falsified. an untested invariance means inconclusive. a valid attack that changes a decision means challenged.

some limits. one task family, mlp blocks only, two small models. attention heads, where this task actually lives, come next. the windows pytorch wheel ships without flash attention, so that kernel shows up as not run on windows. and none of this says anything about model safety.

mit license.
