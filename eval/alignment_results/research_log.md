# PlasmidLM Research Log

---

## Entry

**Date:** 2026-03-10

**Project(s):** PlasmidLM

**Tags:** scorers, curriculum-design, planning

### What I did

Added `ValidPlasmidScorer` with structural penalties for curriculum learning. Also started planning binary motif curriculum approach: the plannotate/motif scorers give partial credit which makes GRPO learning slow. Analyzed the training data to understand hard token distribution.

### What I found / decided

- `training_pairs_v4.parquet` has 108K rows, 57 unique hard tokens, most prompts have 4-13 hard tokens
- Only 51 existing single-token prompts in the dataset -- not enough for single-token curriculum stage
- Decision: build a data generation script that explodes each row into one row per hard token, creating synthetic single-token prompts paired with the original sequence. This gives the model a clear learning signal (one motif at a time) before scaling to multi-token.
- `BinaryMotifScorer` (already in repo) is ideal: hard 0/1 reward forces the model to actually produce the motif rather than getting partial credit for approximate matches.

### Open questions

- How many curriculum stages? 1-tok -> 2-tok -> ... -> full prompt?
- Will BLAST scoring be the bottleneck at scale?

### Next

Implement data generation script and GRPO config for single-token binary motif training.

---

## Entry

**Date:** 2026-03-13

**Project(s):** PlasmidLM

**Tags:** GRPO, binary-motif, curriculum, Anyscale, infrastructure

### What I did

Built and launched the single-token binary motif curriculum training pipeline end-to-end:

1. Created `scripts/generate_curriculum_data.py` -- extracts hard tokens from training pairs, generates one row per token with `<BOS><TOKEN><SEQ>` prompts. Supports `--max-tokens N` for future curriculum stages.
2. Generated data: 221,772 rows across 57 unique single-token prompts (after dedup). Top tokens: `<ELEM_TRACRRNA>` (22K), `<ELEM_IRES>` (22K), `<ORI_COLE1>` (21K).
3. Created `post_training/configs/grpo_binary_motif_anyscale.py` -- binary_motif scorer, 8 generations, temp=0.3, kl_coef=0.3, 1000 steps.
4. Uploaded data to S3, submitted Anyscale job on g6e.xlarge (L40S).
5. Attempted to also run on p5.4xlarge (H100) -- hit `InsufficientInstanceCapacity` across all us-east-1 AZs, including spot. H100s simply not available.

### What I found / decided

- L40S job ran successfully to completion: 1000 steps, ~46s/step, ~12.8 hours total, ~$20.
- Reward trajectory: started ~0.17 at step 85, climbed to 0.3-0.55 range by step 1000. Final step reward=0.531.
- KL drifted from 0.08 to 0.22 over the run -- moderate but notable.
- **Key issue**: LR cosine-decayed to zero by step 1000, so the model stopped learning in the last ~100 steps. The plateau may be partly artificial.
- Checkpoints saved every 100 steps to S3 + final checkpoint.

**Links:**
- W&B run (v1): https://wandb.ai/<entity>/PlasmidLLM/runs/apb93zow
- Anyscale job: https://console.anyscale.com/jobs/prodjob_hqrkwgqiualwbwjk91zvgdst99
- S3 checkpoints: `s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/checkpoints/grpo_binary_motif/`

### Open questions

- Is the 0.5 reward plateau real or just LR starvation?
- Should we push for higher reward on single-token before moving to multi-token curriculum?

### Next

Resume from v1 final checkpoint with tighter KL, lower temperature, and a fresh 2000-step LR schedule.

---

## Entry

**Date:** 2026-03-16

**Project(s):** PlasmidLM

**Tags:** GRPO, binary-motif, debugging, CUDA-graphs, hyperparameters

### What I did

Designed and launched v2 of the binary motif curriculum run, resuming from v1's final checkpoint. Key hyperparameter changes:

| Param | v1 | v2 | Rationale |
|-------|----|----|-----------|
| kl_coef | 0.3 | 1.0 | Tight clamp -- model already knows motifs, prevent drift |
| temperature | 0.3 | 0.2 | Exploit mode -- BLAST needs precise sequences |
| cliprange | 0.2 | 0.15 | Tighter clipping for stability |
| LR | 5e-6 | 3e-6 | Lower peak since we're refining, not learning from scratch |
| steps | 1000 | 2000 | Full schedule so LR doesn't die early |

Hit two issues during launch:

1. **CUDA graphs error** (`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`). This didn't happen in v1. Appears to be PyTorch auto-enabling CUDA graphs, and the generate/train phases use different batch shapes that invalidate the graph cache. Fix: set `TORCH_CUDAGRAPHS=0` and `TORCHDYNAMO_DISABLE=1` as env vars.

2. **p5.4xlarge still unavailable** across all us-east-1 AZs, even spot. Submitted both H100 spot and L40S jobs -- L40S is the workhorse.

### What I found / decided

- v2 is training successfully on L40S. Early results very promising: **reward=0.672 at step 4** (v1 peaked at 0.531 after 1000 steps). The tight KL (0.0008 at step 4 vs v1's 0.22) is keeping the model focused.
- ~37s/step on L40S -> 2000 steps ~ 20.5 hours, ~$33 total.
- Confirmed that `enable_cross_zone_scaling: true` in Anyscale compute config does cycle through AZs (saw attempts in 1b, 1d, etc.), but p5 capacity is genuinely scarce in us-east-1 right now.

**Links:**
- Anyscale job (v2 L40S): https://console.anyscale.com/jobs/prodjob_i5ymtkxjs9i17el39tijf3nln2
- W&B run (v2): project https://wandb.ai/<entity>/PlasmidLLM, run name `grpo_binary_motif_1tok_v2`
- S3 checkpoints (v2): `s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/checkpoints/grpo_binary_motif_v2/`

### Open questions

- Will the high early reward sustain, or is it an artifact of the warmup phase (LR still ramping)?
- At what reward threshold should we move to 2-token curriculum? 0.8? 0.9?
- CUDA graphs issue: is this a torch version regression on the Anyscale image, or has it always been latent? Disabling dynamo costs some speed but not critical for this model size.

### Next

- Monitor v2 run to completion (~20 hours from now).
- If reward > 0.8, generate 2-token curriculum data (`--max-tokens 2`) and plan next stage.
- Consider uploading best checkpoint to HuggingFace as `<org>/PlasmidLM-kmer6-binary-motif-1tok`.
