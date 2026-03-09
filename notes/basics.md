# Basics

## Training data location

All data lives in a single S3 bucket:

```
s3://phd-research-storage-1758274488/databricks_export/
```

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

### How the training data was built

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

The motif registries are built separately by `scripts/build_motif_registry.py`, which extracts canonical reference sequences from plannotate's BLAST/Diamond databases for each categorical token.

### Downloading

```bash
# Current training set + motif registry
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet data/
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/motif_registry_combined.parquet data/

# Everything
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/ data/s3_export/ --recursive
```
