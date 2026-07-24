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
