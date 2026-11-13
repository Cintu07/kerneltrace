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

## bf16 can decide what patching tells you

that last result got its own audit. four models, 48 ioi prompts each, and the same mlp patching sweep run in fp32, in bf16, in bf16 with only the unembedding in fp32, and in bf16 with attention and the unembedding in fp32. fp32 is the reference.

![one attention head in bf16](docs/figures/x_bf16_head.png)

pythia-160m is the bad one. on 25 of the 48 prompts the clean and corrupted logit diffs come out identical in bf16, so there's no effect left to patch. its answer logits sit near 836, and bf16 only has values every 4 out there. on the prompts that survive, bf16 says mlp layer 4 brings back 43% of the answer. fp32 says 3%.

an fp32 head doesn't save it. the damage is inside attention. from layer 3 on, pythia-160m adds a huge offset to every attention score of a query, the same for every key, up to 115,448 in layer 11. softmax subtracts it back out, so in exact math it does nothing. but the part that's left, the spread between keys that decides where a head looks, is 20 to 100. bf16 steps by 256 or 512 at that size. in the head above, every score rounds to 35,584 and attention that was 86% on one word turns into 7% on all of them.

![what bf16 does to patching](docs/figures/x_bf16_audit.png)

running attention in fp32 and the rest in bf16 fixes it on all four models. median error drops to 0.003 to 0.007, and the top 3 layers match fp32 on every pythia model. on gpt-2 the third place swaps between layers 9 and 11, which tie at 0.031 in fp32 anyway. pythia-410m has the same offsets (up to 36,654) and loses its layer 5 effect in bf16. pythia-1b doesn't have them (largest is 52) and bf16 is fine there, so it isn't about size. `scripts/attention_offsets.py` measures the offsets before you run anything. gpt-2 small has a smaller problem in its logits, which bf16 rounds to steps of 0.5, and an fp32 head alone handles that.

```python
from kerneltrace.models.precision import upcast_attention
adapter = upcast_attention(load(model_id, revision, "eager", "bfloat16", "cuda"))
logits, _ = run(adapter, ids, patches=..., fp32_logits=True)
```

the predictions were written down and committed before the run, in `experiments/bf16_audit_prediction.txt`. two held and four didn't. fp32 wasn't perfectly batch-stable on pythia-160m, which drifts by up to 0.007 because its logits are so big. gpt-2 jumped by 0.5 on 10 prompts, short of the quarter i guessed. pythia's jumps weren't bigger either. on most prompts neither model moved, and fewer pythia-160m prompts jumped than gpt-2 prompts, 5 against 12, because a grid of 4 swallows small drift and then lets go by 4 or 8. and the fp32 head alone didn't fix pythia, because the problem was never the head.

## where the offsets come from

the key bias. in the dimensions rotary embeddings leave alone, a head's key bias adds q·b to the score of every key a query can see, the same number for every key. softmax subtracts it, so it changes nothing, and in exact math it gets no gradient. in pythia-160m it starts at zero and grows to a norm of 279 over training. the part rotary does touch, which the loss can see, stays under 20. reading the final weights of the whole pythia suite with range requests, 70m, 160m, 410m and 1.4b all carry one over 100.

![the key bias](docs/figures/x_key_bias.png)

## why it grows

the cerebras-gpt paper (2023, appendix a.2) already noticed key biases growing fast in fp16 training and blamed underflow in the softmax gradients. it didn't test that. so i trained a small pythia-shaped model from scratch, 4 layers with rotary on a quarter of each head, the same seed and data every time, and changed only the precision or the optimizer.

![why it grows](docs/figures/x_key_bias_training.png)

in fp32 the invariant part of the key bias ends at 0.004 after 3,000 steps. in fp16 it's 0.340, and in bf16 0.504. the loss curves of the three runs are the same to three decimals, so the models learn the same thing. the difference is all in the part softmax can't see.

the gradient explains it. the invariant part's gradient is the row sum of the softmax score gradient times q, and that sum is zero in exact math. in fp16 the score gradient gets cast down before the matmul, the sum stops being zero, and the invariant part gets a gradient about 2,600 times smaller than the rotary part's. adam divides each step by the gradient's own recent size, so that tiny gradient gets a step 1.16 times as big as the rotary part's. in fp32 the gradient is another 800 times smaller, it sits under adam's eps of 1e-8, and the step comes out near zero.

two controls pin it down. fp16 everywhere except attention, which runs in fp32, ends at 0.004, the same as full fp32. so the leak is in the attention backward. adam with eps 1e-4 in fp16 ends at 0.000, but it also slows every other small weight, and the rotary part only reaches 2.18 instead of 4.04. the run doesn't show where the sign of the leak comes from, or how far it would have gone over pythia's 143,000 steps.

nine predictions, in `experiments/key_bias_training_prediction.txt`, the ninth added before its run. seven held. the invariant part never got bigger than the rotary part in 3,000 steps (0.34 against 4.04), and i'd guessed it would be twice as big. the fp16 gradient ratio was 4e-4, not above 1e-2. the sgd run moved no bias at all, rotary part included, so it says nothing about adam and the eps run carries that instead.

## outside pythia

any model with a key bias has the invariant part. with learned positions or alibi it's the whole key bias, with partial rotary it's the dims rotary skips. `scripts/key_bias_families.py` reads it out of 23 checkpoints from eight families the same way as the pythia suite, bias tensors only.

![outside pythia](docs/figures/x_key_bias_families.png)

it grows in some of them. santacoder's is 384 and bloom-560m's 183, against query biases of 7 and 11. the rest of bloom, phi, stablelm-7b and gpt-neox-20b sit between 14 and 80. gpt-2 and opt stay under 10, and gpt-2 medium, large and xl are below their own query bias, so it isn't something every model does.

qwen2.5 has rotary on every dim, so none of its key bias is exactly invariant. but its rotary base is a million, and its slowest dims turn less than a tenth of a radian over 4,096 positions. that's where its key bias sits, 283 in 0.5b, 821 in 1.5b and 920 in 7b, against 35 to 120 in all the other dims. zeroing it there moves the fp32 logits by 0.18 and the perplexity by under 0.01%.

big doesn't mean broken. `scripts/key_bias_other_models.py` runs perplexity with and without it. santacoder gives nan on every window in fp16, and with the invariant part dropped fp16 matches fp32. that edit moves the fp32 logits by 0.0015. but its bf16 is only 0.1% off. bloom-560m loses 13.4% in bf16 and dropping its 183 changes nothing. that one is its output logits, below. qwen and opt are fine in both.

twelve predictions in `experiments/key_bias_families_prediction.txt`, five held. i expected the growth to be more common and to do more damage than it does. opt, bloom-7b1, gpt-neox-20b and qwen-1.5b all came in under what i guessed, santacoder broke in fp16 and not bf16, bloom's bf16 loss wasn't the bias, and qwen didn't overflow fp16. cerebras-gpt is gated now and wasn't checked.

it reaches past patching. on wikitext-2, pythia-70m's perplexity goes from 68 to 659 in bf16. 160m gets 171% worse, 410m 20%, 1.4b almost 6%. in fp16, 70m, 160m, 410m and 1.4b give nan on every window.

zeroing that part of the bias is an exact edit, `drop_invariant_key_bias(adapter)`. in fp32 the logits move by at most 0.007. after it, pythia-410m runs in plain bf16 within 0.1% of fp32 and its patching sweep matches fp32, and fp16 is finite again on 70m, 160m and 410m. it doesn't cover everything. 70m and 160m also need the fp32 head, because their output logits are huge. 1.4b still overflows fp16, because hf multiplies q by k before scaling and that product hits 129,160 even without the bias. keeping attention and the head in fp32 handles every case here, within 0.6% of fp32 perplexity on all six models.

the predictions for this part are in `experiments/key_bias_prediction.txt`, each batch committed before it ran. six held and three didn't. plain bf16 with the bias dropped wasn't enough for pythia-160m's patching sweep (median error 0.051, i'd said 0.02). the 25 prompts where bf16 saw no effect came back with the bias alone, when i'd guessed the output grid would keep more than 10 of them lost. and dropping the bias left 160m's bf16 perplexity 60% off instead of within 0.5%, which is how the second problem, its huge output logits, showed up.

## the interaction floor

baligar (2026, computation 14(9):212) measured pair interactions, q = f11 - f10 - f01 + f00 from four patched runs, in fp32 and fp64 on gpt-2 medium and pythia-410m-deduped. pythia's fp32 error came out ten times gpt-2's and the paper doesn't say why. pythia-410m carries an invariant key bias of 132, so `scripts/patching_floor.py` reruns that kind of measurement on ioi with and without it: residual stream patches at 4 depths and 2 positions, 28 pairs per prompt, 20 prompts, 560 pairs per model. the fp64 run keeps softmax and rotary in fp64. transformers casts both to fp32 for gpt-neox, which baligar also ran into.

![the interaction floor](docs/figures/x_patching_floor.png)

the floor (median |q32 - q64|) on pythia-410m-deduped is 3.0e-4 as released and 9.7e-6 with the bias dropped, 31 times lower. on pythia-410m it goes from 1.7e-4 to 3.3e-6, 52 times. gpt-2 medium, whose invariant part is 2.6 per head, sits at 9.2e-6 both ways. so with the bias gone the two pythias land at or under gpt-2 medium, and the 10x gap is the bias. in fp64 the two versions of each model agree to 1e-11, so nothing about the model changed, only the rounding.

baligar also found that transformers' gpt-neox softmax, which casts scores to fp32 even in an fp64 model, dominates the error of nominal fp64 runs. that cast only costs much when the scores carry a big offset, and they do: on these prompts pythia-410m-deduped's largest score offset is 55,579 against a spread of 17 between keys. `scripts/fp64_softmax_cast.py` runs fp64 with only the stock softmax. the cast alone gives an interaction error of 1.1e-4 on 410m-deduped, over a third of its whole fp32 floor. with the bias dropped it falls 44 times, and 235 times on pythia-410m.

bf16, which baligar didn't test, is much worse. the bf16 floor is 0.06 to 1.4, the size of the interactions themselves, and on pythia-410m-deduped 71% of pairs have an error bigger than the interaction. dropping the bias takes that to 46%. interactions aren't readable in bf16 on any of the three models.

## a second task

the patching audit was ioi only. capital cities is the second family: "the capital of france is the city of" against the same prompt with germany, 4 templates, 48 prompts each model gets right. it's recall, which leans on mlps where ioi leans on attention.

the picture holds. pythia-410m's median error is 0.024 in bf16 and 0.002 with the bias dropped. pythia-160m needs the drop and an fp32 head together (0.053 to 0.004), because its capital logits sit near 830 where bf16 steps by 4, and either fix alone leaves most of the error. pythia-1b and gpt-2 stay under 0.013 in plain bf16, as they did on ioi, since neither has offsets big enough for bf16 to round away the spread between keys.

## with fused attention

everything above used the plain eager attention, which rounds the scores to bf16 before softmax. pytorch's fused attention (sdpa, the memory-efficient kernel, since this windows build has no flash attention) keeps the scores and softmax in fp32, but each key is still stored in bf16 with the bias already added. `python scripts/bf16_audit.py 48 pythia-160m pythia-410m --attn sdpa` reruns the ioi audit with it. pythia-160m's median error is 0.060 against 0.098 eager, and only 2 prompts lose their effect instead of 25. pythia-410m's is 0.027 against 0.107. dropping the bias still takes 410m to 0.007. so the fused kernel removes a lot of it and not all of it. `scripts/bootstrap_ci.py` puts 95% intervals over prompts on these and on the floors, and none of the comparisons above overlap.

## the offsets themselves, and bloom

`scripts/score_offsets.py` catches every attention score row at softmax in fp32, on 4 windows of 512 tokens, and checks what bf16 rounding of those scores alone does to the attention. on pythia-160m it moves 45.6% of rows by more than 0.1 in total variation, and 0.15% with the bias dropped. santacoder's raw q·k reaches 148,785, more than twice fp16's 65,504, which is the nan. dropped, it peaks at 1,579.

bloom-560m's 13.4% bf16 loss turned out to be something else. dropping the bias cut its largest offset from 5,848 to 191 and didn't move the perplexity. writing its alibi as slope·(key - query), which removes another shared offset that transformers adds, didn't either (+12.9%). fp16 is fine (+0.2%), so it's bf16's short significand somewhere. `scripts/bloom_bisect.py` keeps one group of modules in fp32 at a time: attention, mlps and layernorms change nothing, the head alone takes it to +0.6%. its logits sit near 430, where bf16 steps by 2.

twenty-one predictions for this part, in `experiments/followups_prediction.txt`, all committed before their runs. fifteen held, including all three about the softmax cast. i expected fused attention to keep at least half of the bf16 error on both pythias, and on 410m it kept a quarter. santacoder moves 3.5% of its attention rows in bf16, not under 1%, and still loses only 0.1% perplexity. bloom's offsets did come from the bias, i guessed they didn't. dropping the bias cut bf16's bad pairs on 410m-deduped from 71% to 46%, short of half. capitals on 410m showed 0.024 of bf16 error, not 0.05. and the alibi rewrite didn't fix bloom, which is what sent me to the bisect that found its head.

## run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the build for your gpu
pip install -e ".[dev]"
pytest -q                                          # 71 tests, cpu only, no downloads
kerneltrace run experiments/gpt2_ioi_mlp0.yaml     # about 6 minutes on an 8 gb laptop gpu
kerneltrace replay evidence/<bundle>               # rerun a bundle and compare it to what was stored
python scripts/causal_trace.py 24                  # the traces behind the picture above
python scripts/bf16_batch_check.py                 # one prompt, two batch sizes, the 0.5 jump
python scripts/bf16_audit.py && python scripts/bf16_summary.py   # the bf16 audit, about 15 minutes
python scripts/key_bias.py                         # parity of the bias edit, training checkpoints, the whole suite
python scripts/lm_quality.py                       # wikitext-2 perplexity in fp32, bf16 and fp16
python scripts/key_bias_training.py && python scripts/key_bias_training_summary.py   # train it six ways, about 30 minutes
python scripts/key_bias_families.py                # the invariant key bias in 23 checkpoints outside pythia
python scripts/key_bias_other_models.py            # perplexity with and without it, santacoder, bloom, qwen, opt
python scripts/score_offsets.py                    # score offsets at softmax, and what bf16 rounding does to them
python scripts/patching_floor.py 20                # fp32 and bf16 interaction floors against fp64, about 45 minutes
python scripts/bf16_audit.py 48 --task capitals    # the patching audit on the second task
python scripts/bloom_alibi.py && python scripts/bloom_bisect.py   # what bloom's bf16 loss is and isn't
python scripts/fp64_softmax_cast.py                # how much transformers' fp32 softmax costs an fp64 run, with and without the bias
python scripts/followups_summary.py                # scores predictions 1-16
```

a run writes a folder under `runs/`. inside is the manifest (model commit, library and driver versions, gpu, tf32 and determinism switches, git state), the claim with its final status, every generated prompt with the reason any were rejected, raw per-case results, the kernels the profiler saw, and a report.

## what's in here

`kerneltrace/` is the library: claim cards, the ioi task, gpt-2 and pythia adapters, patching, the three attacks, the stats and the verdict rules. `experiments/` has the claim cards and the specs that test them. `evidence/` holds the bundles from the runs above. `scripts/` draws the figures and computes the causal traces.

the status rules live in `kerneltrace/core/verdict.py`. short version: a failed gate means inconclusive. an interval on the wrong side of its threshold means falsified. an untested invariance means inconclusive. a valid attack that changes a decision means challenged.

some limits. two task families, mlp blocks and the residual stream only, small models. the claim cards use two, the bf16 audit goes up to pythia-1b, perplexity up to 1.4b, and the bias survey reads pythia up to 12b and 23 other checkpoints up to 20b without running them. attention heads, where this task actually lives, come next. the windows pytorch wheel ships without flash attention, so that kernel shows up as not run on windows. and none of this says anything about model safety.

mit license.
