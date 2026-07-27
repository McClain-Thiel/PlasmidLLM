# Data

Git contains only the vocabulary files needed to define the tokenizer:

| file | purpose |
|---|---|
| `special_tokens.txt` | Categorical and control tokens, one per line |
| `token_vocabulary_v4.json` | Token-to-ID mapping used by the released models |

All Parquet datasets, motif registries, generated panels, and model checkpoints
are external artifacts. They are ignored intentionally; do not add them to Git.

## Expected local layout

Place downloaded inputs in this directory:

```text
data/
├── training_pairs_v4.parquet
├── full_sequences.parquet
├── metadata.parquet
├── annotations.parquet
├── motif_registry.parquet
├── motif_registry_combined.parquet
└── plannotate_db.parquet
```

The main inputs are:

| file | consumer |
|---|---|
| `training_pairs_v4.parquet` | Pretraining and GRPO prompt sampling |
| `full_sequences.parquet` | Source-plasmid holdout construction |
| `metadata.parquet` | Plasmid metadata joins |
| `annotations.parquet` | Historical pLAnnotate annotations |
| `motif_registry*.parquet` | Alignment rewards and evaluator bridges |
| `plannotate_db.parquet` | BLAST-based pLAnnotate scorer |

The complete NeurIPS rebuttal snapshot—including exact inputs, generated
sequences, raw BLAST output, checkpoints, manifests, and source bundles—is
indexed in [`infra/DATA_LOCATIONS.md`](../infra/DATA_LOCATIONS.md). The public
review branch uses an anonymized bucket placeholder.

## Training-pair schema

The pretraining loader accepts either a `full_text` column or paired
prompt/sequence columns. A formatted row has this form:

```text
<BOS><categorical tokens><SEP>DNA<EOS>
```

Prompt-token arrays and source plasmid IDs should be retained when constructing
new datasets so train/evaluation overlap can be audited.

## Rebuilding the motif registry

```bash
PLASMID_ANNOTATION_ROOT=/path/to/addgene_clean \
  python scripts/build_motif_registry.py
```

The builder also requires the pLAnnotate BLAST/Diamond/Infernal databases and
their corresponding command-line tools.
