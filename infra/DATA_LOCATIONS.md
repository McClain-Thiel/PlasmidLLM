# Data, Archive, and Paper Provenance Index

Large Parquet datasets, evaluation bundles, and model checkpoints are gitignored; canonical copies are hosted on Hugging Face and verified by SHA-256 checksums.

## Hugging Face Model Repositories & Storage Buckets

| Resource | URI | Contents |
|---|---|---|
| Pretrained Base (`v2.0` / `v1.0`) | `McClain/PlasmidLM-kmer6` | `main`/`v2.0` (`cluster_base_s42`, 100k steps); `v1` branch (`v1.0`, 65k steps) |
| Post-GRPO (`v2.0` / `v1.0` & ablations) | `McClain/PlasmidLM-kmer6-GRPO-plannotate` | `main`/`v2.0` (`exp2_v2_s44`); `v2-seed{42,43,44}`, `exp1-v1-seed{42,43,44}`, `exp1-v2-seed{42,43,44}` |
| Primary Datasets Bucket | `hf://buckets/McClain/PlasmidRL/datasets/` | `cluster_v3/` (certified split) and `v4/` (source pairs & motif/pLannotate DBs) |
| Rigor & Rebuttal Archive Bucket | `hf://buckets/McClain/PlasmidLMEval/` | `rigor_20261116/` (cluster-disjoint paper study) and `neurips_2026_rebuttal/` (legacy source-ID archive) |

Training telemetry is logged at `https://wandb.ai/mcclain/PlasmidLLM`.

## Canonical Data Files & Checksums

| Local Path | Rows | Size | SHA-256 |
|---|---:|---:|---|
| `data/special_tokens.txt` | 106 lines | 1.3K | `bbd63ec88d035134d2ef9b3af54b54382bb093c334043266ba36e76dd6322d41` |
| `data/token_vocabulary_v4.json` | 106 keys | 2.3K | `5f76f8c805065e867ee6242cbc86989c8c3ac37d03c20239b288727007a53193` |
| `data/training_pairs_v4.parquet` | 108,468 | 690M | `93beca93c675bcb979ee4046911accda5536f07731cd5cc41e5a2d7ab2807c69` |
| `data/cluster_v3/train.parquet` | 93,900 | 602M | `f299ea8ccbf6941b33b81d4c6bff3ad775688e2fedea3799f4c93081848ebc4d` |
| `data/cluster_v3/validation.parquet` | 4,728 | 27M | `bc562467415aa1d28b7db867b0a376d02690dc86f0d735401ea0139733e42227` |
| `data/cluster_v3/test.parquet` | 4,939 | 25M | `6b4f4169512de13295f50f3e393836712d752be2c245dd4ca1da1176c55cdfc0` |
| `data/cluster_v3/assignments.parquet` | 103,567 | 5.3M | `0d8b2d7d67c3a4512f4c2acefea1d670ea26bc51d4d9dda0d8011a23c896a873` |
| `data/cluster_v3/manifest.json` | — | 2.3K | `5cd70a45bf316457f4fba4e60a3874aa847b6d56faff95ecf91737551d2eda91` |
| `data/cluster_v3/audit/report.json` | — | 6.8K | `3fa1fb07cbdf9b3d65009fc6cf5f34fe1eb12c09f4f650bb0b1c3614bfafa17b` |
| `data/motif_registry_combined.parquet` | 660 | 138K | `b8431d37bfedb77e982bce775022d285c45c90ed6c9a95ffc9df8cc767626493` |
| `data/plannotate_db.parquet` | 567,340 | 177M | `48ca30df2abbb953ddeed9da20d2f6af9a5c57d136865231b47fadea2b12f85d` |
| `data/holdout_full_prompts.parquet` | 22,400 | 61M | `7ee544e18e6ca43d973a711202eb35a68068b8d605eb800a792dff7a5f62322f` |
| `eval/runs/rigor_20261116/inputs/frozen_panel_v3.parquet` | 469 | 3.8M | `c248dc2e9e03a0b479c48af78626800dcf42f2d42cbb31a3f08e53e25e6edb2d` |

Complete file-level verification (`955` files under `eval/runs/rigor_20261116/`) is tracked in [`eval/runs/rigor_20261116/SHA256SUMS`](../eval/runs/rigor_20261116/README.md).

## Manuscript Table & Figure Provenance Map

All primary paper results originate from `eval/runs/rigor_20261116/` (`frozen_panel_v3`, `468` main clusters):

| Manuscript Item | Generating Script(s) | Primary Source Artifact(s) |
|---|---|---|
| Split & Audit (`Sec. 3.1`) | `pretraining/{build,refine,verify}_cluster_split.py` | `data/cluster_v3/{manifest.json,audit/report.json}` |
| **Table 1** (Primary & BoK) | `experiments/neurips_rebuttal/summarize_rigor_evaluation.py`, `manuscript_diagnostics.py` | `results_v1/{summary.json,manuscript_diagnostics.json}` |
| **Table 2** (External baselines) | `eval/scripts/score_cluster_v3_baselines.py` | `baselines_v1/external_baselines_summary.{csv,json}` |
| **Table 3** (Categories) | `summarize_rigor_evaluation.py` | `results_v1/tables/category_and_tag_breakdown.csv` |
| **Table 4** (Hyperparameters) | `pretraining/configs/cluster_disjoint_v2.py`, `post_training/configs/grpo_reward_ablation.py` | `checkpoints/rigor_20261116/*/config.json` |
| **Table 5** (3-seed & BoK) | `summarize_rigor_evaluation.py`, `manuscript_diagnostics.py` | `results_v1/{summary.json,tables/paired_comparisons.csv,manuscript_diagnostics.json}` |
| **Table 6** (Threshold & clean-prompt) | `summarize_rigor_evaluation.py`, `manuscript_diagnostics.py` | `results_v1/{all_observations.parquet,manuscript_diagnostics.json}` |
| **Table 7** (Per-token AMR-11) | `summarize_rigor_evaluation.py`, `manuscript_diagnostics.py` | `results_v1/{tables/per_token_breakdown.csv,manuscript_diagnostics.json}` |
| **Table 8** (Minority/`COPY_LOW`) | `summarize_rigor_evaluation.py` | `results_v1/tables/minority_panels_summary.csv` |
| **Table 9** (`EOS`/Seen/Novelty) | `manuscript_diagnostics.py` | `results_v1/{manuscript_diagnostics.json,sequence_novelty_summary.json}` |
| **Virulence screen** (`Sec. 6`) | `manuscript_diagnostics.py` | `annotations/{reference_control,cluster_base_s42,exp2_v2_s42..44}/*/amrfinder.tsv` |
| **Figures 1 & 3–9** | `eval/scripts/generate_figures.py` | `results_v1/{all_observations.parquet,summary.json,tables/*.csv}` + `annotations/exp2_v2_s44/decode_s42/plannotate_results.json` |
