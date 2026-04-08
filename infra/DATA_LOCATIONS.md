# Data Locations

## S3 — Primary Training Data

**Bucket:** `s3://phd-research-storage-1758274488/databricks_export/`

| File | Size | Description |
|------|------|-------------|
| `training_pairs_v4.parquet` | 689.9 MiB | Current training set (v4 vocabulary, 106 tokens) |
| `training_pairs.parquet` | 679.8 MiB | Previous training set (v3 vocabulary) |
| `full_sequences.parquet` | 401.0 MiB | Raw plasmid sequences from Addgene |
| `annotations.parquet` | 233.0 MiB | Plannotate annotations for each plasmid |
| `partial_sequences.parquet` | 47.3 MiB | Partial/incomplete sequences |
| `metadata.parquet` | 24.5 MiB | Addgene metadata (species, vector type, etc.) |
| `motif_registry_combined.parquet` | 137.6 KiB | Extended motif registry (multiple DB sources) |
| `motif_registry.parquet` | 54.5 KiB | Original motif registry |

### Download

```bash
# Current training set + motif registry
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet data/
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/motif_registry_combined.parquet data/

# Everything
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/ data/s3_export/ --recursive
```

### How Training Data Was Built

```
full_sequences.parquet    raw DNA from Addgene
        +
annotations.parquet       plannotate feature calls (AMR, ORI, PROM, ...)
        +
metadata.parquet          species, vector type, copy number
        ↓
  (notebooks/Build_Training_Pairs.py)
        ↓
training_pairs_v4.parquet   <BOS><tokens><SEP>SEQUENCE<EOS> formatted pairs
```

Motif registries are built by `scripts/build_motif_registry.py` from plannotate's BLAST/Diamond databases.

## S3 — Anyscale Training

**Bucket:** `s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/`

```
├── data/
│   ├── motif_registry_combined.parquet
│   └── training_pairs_v4.parquet
└── checkpoints/
    └── grpo_dense_motif/
        ├── step_100/
        └── final/
```

### Download Checkpoint

```bash
aws s3 sync \
  s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/checkpoints/grpo_dense_motif/final/ \
  ./checkpoints/grpo_dense_motif_anyscale/
```

## Local — This Machine

- `/opt/dlami/nvme/PlasmidLLM/` — stale copy of repo (on `reorg` branch, older than `~/PlasmidLLM`)
- `/opt/dlami/nvme/plasmid_data_0_raw/` — raw data
- `/opt/dlami/nvme/plasmid_data_1_parsed/` — parsed data
- `/opt/dlami/nvme/plasmidkit_addgene_annotations/` — AddGene annotations
- `/opt/dlami/nvme/plasmidkit_annotations/` — additional annotations

## HuggingFace Hub

- `McClain/PlasmidLM-kmer6` — 19.3M param dense transformer
- `McClain/PlasmidLM-kmer6-MoE` — MoE variant

## Monitoring

- WandB: https://wandb.ai/mcclain/PlasmidLLM
