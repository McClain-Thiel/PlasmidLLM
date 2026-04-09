# PlasmidLLM — Claude Code Instructions

## Storage

- **Root disk** (`/`): 193GB, often near-full. Do NOT use for large downloads, builds, or temp files.
- **NVMe drive** (`/opt/dlami/nvme/`): 549GB mounted ephemeral storage. Use for:
  - Conda environments (`/opt/dlami/nvme/envs/`)
  - Large builds (set `TMPDIR=/opt/dlami/nvme/tmp`)
  - Model checkpoints and large data
- **S3 mount** (`/mnt/s3/phd-research-storage-1758274488/`): Read-only S3 bucket with training data, annotations, and exports. Key paths:
  - `databricks_export/` — training pairs, annotations, motif registries
  - `checkpoints/` — model checkpoints

## Conda Environments

- `plannotate` — bio tools: plannotate 1.2.2, prodigal, blast, sourmash, minimap2, lightgbm, shap, ViennaRNA
- `evo2` — (on nvme) for Evo2 inference: torch 2.8 + flash-attn prebuilt wheel

## Models

- `McClain/PlasmidLM-kmer6-GRPO-plannotate` — main eval target (19.3M params)
- Custom `PlasmidKmerTokenizer` doesn't expose pad/eos/bos tokens — set them explicitly

## Eval Suite

All eval code lives in `eval/`. Data goes to HF bucket `McClain/PlasmidLMEval` (use `hf buckets` CLI, NOT Python API which needs huggingface-hub>=1.0 incompatible with transformers).

## HuggingFace

- Authenticated as McClain (write access)
- `huggingface-hub` pinned to <1.0 for transformers compat
- For bucket operations use plannotate env which has hf-hub 1.x, or use `hf buckets` CLI
