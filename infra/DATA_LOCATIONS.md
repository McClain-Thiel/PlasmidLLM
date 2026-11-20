# Data and artifact layout

Large datasets, generated panels, and checkpoints are not stored in Git; canonical copies are hosted on Hugging Face (`McClain/PlasmidLMEval`, `McClain/PlasmidRL`, and model repos below).

## Training inputs

| file | purpose |
|---|---|
| `training_pairs_v4.parquet` | Pretraining pairs in `<BOS><tokens><SEP>sequence<EOS>` format |
| `full_sequences.parquet` | Larger source-plasmid table used to construct source-ID holdouts |
| `metadata.parquet` | Copy number and other plasmid metadata |
| `annotations.parquet` | Historical pLAnnotate export; incomplete and not suitable for rebuilding every prompt |
| `motif_registry_combined.parquet` | Prompt-token to reference-sequence registry used by post-training |

The local convention is:

```text
data/
├── training_pairs_v4.parquet
├── full_sequences.parquet
├── metadata.parquet
├── annotations.parquet
└── motif_registry_combined.parquet
```

Only `special_tokens.txt` and `token_vocabulary_v4.json` are tracked. All other files under `data/` are ignored. These primary datasets and the BLAST-certified `cluster_v3` split live at:

```text
hf://buckets/McClain/PlasmidRL/datasets/
├── cluster_v3/   # train.parquet, validation.parquet, test.parquet, assignments.parquet, manifest.json, audit/report.json
└── v4/           # training_pairs_v4.parquet, plannotate_db.parquet, motif_registry_combined.parquet, holdout_full_prompts.parquet, token_vocabulary_v4.json, special_tokens.txt
```

## Evaluation & Reproducibility Archives

- **Three-seed cluster-disjoint & reward-ablation archive (`frozen_panel_v3`):**
  `hf://buckets/McClain/PlasmidLMEval/rigor_20261116/` (`inputs/`, `evaluations_v1/`, `annotations/`, `results_v1/`, `split/cluster_v3/`, `checkpoints/`, `SHA256SUMS`).
- **Source-ID-disjoint & rebuttal snapshot:**
  `hf://buckets/McClain/PlasmidLMEval/neurips_2026_rebuttal/` (indexed by `lineage/SHA256SUMS`; see [`experiments/neurips_rebuttal/README.md`](../experiments/neurips_rebuttal/README.md)).

## Model repositories

- `McClain/PlasmidLM-kmer6` — Pretrained 19.3M-parameter base model (`main` / `v2.0`: `cluster_base_s42` 100k-step cluster-disjoint base; `v1` / `v1.0`: original 65k-step base)
- `McClain/PlasmidLM-kmer6-GRPO-plannotate` — Post-trained GRPO checkpoint (`main` / `v2.0`: `exp2_v2_s44`; `v2-seed{42,43,44}`, `exp1-v1-seed{42,43,44}`, `exp1-v2-seed{42,43,44}` branches; `v1` / `v1.0`: original source-ID GRPO checkpoint)

Training and evaluation runs are logged at https://wandb.ai/mcclain/PlasmidLLM.
