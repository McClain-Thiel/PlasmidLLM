# Data and artifact layout

Large datasets, generated panels, and checkpoints are not stored in Git. The anonymized
review branch uses placeholder namespaces; substitute the archive location supplied by the
authors.

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

Only `special_tokens.txt` and `token_vocabulary_v4.json` are tracked. All other files under
`data/` are ignored.

## Rebuttal archive

The complete external snapshot is rooted at:

```text
hf://buckets/<org>/PlasmidLMEval/neurips_2026_rebuttal/
```

Its `README.md` and SHA-256 manifest index source bundles, inputs, generated sequences,
annotations, logs, W&B lineage, and checkpoint binaries. Within the snapshot:

```text
neurips_2026_rebuttal/
├── README.md
├── lineage/
│   ├── SHA256SUMS
│   └── local_file_manifest.json
├── source/
└── s3/neurips_rebuttal/
    ├── artifacts/
    ├── m13_blind_v2_g7e/
    ├── grpo_m13_blind_v2_g7e_ckpt/
    └── exhaustive_training_mosaic_20260727/
```

The historical archive intentionally contains failed and superseded attempts. Use
[`experiments/neurips_rebuttal/README.md`](../experiments/neurips_rebuttal/README.md) to
identify the final analyses and their interpretation limits.

## Model repositories

- `<org>/PlasmidLM-kmer6` — released dense base checkpoint
- `<org>/PlasmidLM-kmer6-GRPO-plannotate` — released post-GRPO checkpoint

Training and evaluation runs log to the `PlasmidLLM` W&B project. Exact run links remain
in the private research log during double-blind review.
