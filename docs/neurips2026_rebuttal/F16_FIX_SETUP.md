# F16 fix — experiment setup

Goal: make the model produce a requested component substitute instead of reverting to the
training co-occurrence (F16: substituted-in recovery 4.6%, reverts 40.2%, full-spec 0%).

## Attempt 1 — CFG decoding (FAILED, recorded)
`cfg_generate.py`. No guidance weight improves substituted-in recovery over baseline; higher
w only degrades viability. The "which marker" decision is spread across hundreds of k-mer
tokens, so per-token logit amplification adds sequence noise faster than compositional
control. Decode-time fix does not work.

## Attempt 2 — GRPO on novel-combination rollouts (RUNNING)
Hypothesis: the model reverts because it never saw a novel combination during post-training —
rollout prompts came from training_pairs_v4, where the co-occurring marker always IS the
requested one. Continue GRPO from the released post-GRPO checkpoint on a rollout set that is
50% novel combinations; the existing recall reward then optimizes producing the requested
substitute directly. No reward change in attempt 2 (precision penalty is the follow-up).

- config: `experiments/neurips_rebuttal/grpo_novel_aug_config.py` (runs as
  `post_training/configs/grpo_novel_aug.py` on the box)
- rollout prompts: `s3://.../neurips_rebuttal/derived/rollout_novel_aug.parquet`
  (4k novel + 4k natural, built by `build_rollout_prompts.py`)
- start: `McClain/PlasmidLM-kmer6-GRPO-plannotate`; 400 steps, lr 5e-6, kl 0.5, ~62s/step
- eval set during training: `novel_benchmark.parquet` (watch the F16 metric)
- checkpoints → `s3://phd-research-storage-1758274488/neurips_rebuttal/grpo_novel_ckpt/`
- log on box: `/opt/dlami/nvme/rebuttal/grpo_train.log`

### Env patches to run on g6-big (torch 2.9.1) — box copies only, NOT the repo
The training stack was written for older torch; four fixes were applied to
`/opt/dlami/nvme/rebuttal/code/` and the loaded modeling file, all instances of the same
"inference tensors cannot be saved for backward" incompatibility:
1. `post_training/common/model.py`: clone `output_ids` after `generate_simple`
2. `post_training/common/model.py`: gate `torch.compile(reduce-overhead)` behind
   `PLASMID_COMPILE=1` (CUDA-graph capture trips the inference-tensor check); eager by default
3. `plasmid_llm/models/hf_plasmid_lm/modeling_plasmid_lm.py`: `@torch.inference_mode()` →
   `@torch.no_grad()` on `generate_simple`/`generate_compiled` (this is the loaded copy)
4. same change in the HF-cache copy of the modeling file (belt-and-suspenders)

If reproducing the run on a fresh box, these are needed only on torch>=2.5 with the current
training code. The proper repo fix is to change the two decorators to `no_grad` and make
`torch.compile` opt-in; propose to McClain before touching the tracked repo.

## Evaluation when done
Score the final checkpoint on the novel benchmark with `score_novel.py` (same path as F16),
compare substituted-in recovery / full-spec satisfaction to the current GRPO model, and
re-check the disjoint benchmark (F14) to confirm we did not regress in-distribution.

## Attempt 3 — GRPO on producible-only novel combinations (FAILED, F21)

After F20 showed F16 was confounded by unproducible substitutes, rebuilt rollout + benchmark
to substitute only the 18 producible (high natural-lift) components, and raised group size to
G=16 for exploration (prompt_batch_size=4 to fit the L4's 22GB).

- config: `experiments/neurips_rebuttal/grpo_producible.py` (from the released GRPO ckpt)
- producible substitute list: `s3://.../neurips_rebuttal/derived/producible_substitutes.txt`
- rollout: `derived/rollout_producible.parquet` (6k novel + 4k natural)
- eval benchmark: `derived/novel_producible_benchmark.parquet` (500, substitutes ZEOCIN,
  NEOMYCIN, EF1A, RSV, U6, SV40, PSI, …)
- clean paired eval at step 100 (`derived/{fix100,orig_prod}_plann.json`): the model stops
  reverting to the wrong marker (39.8→31.0%) but does not install the requested substitute
  (8.8→10.0%), so viability drops (62.4→47.0%). Run then diverged (~step 150: reward 0.40,
  KL 0.29, grad-norm 26) and was stopped.
- **Conclusion: the recall reward is the binding constraint** — no specific pull toward the
  one requested substitute among ~9 tokens.
- checkpoints: `s3://.../neurips_rebuttal/artifacts/grpo_producible_ckpt/step_{100,...}/`

## Attempt 4 — rarity-weighted recall reward (IN PROGRESS)

Weight the recall term by token rarity so a requested low-base-rate component dominates the
reward. Changes `post_training/scorers/plannotate.py::_compute_composite`. Approved by McClain.
