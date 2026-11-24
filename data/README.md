# Data

Git tracks only the lightweight tokenizer vocabulary definitions in this directory:

| File | SHA-256 | Description |
|---|---|---|
| `special_tokens.txt` | `bbd63ec88d035134d2ef9b3af54b54382bb093c334043266ba36e76dd6322d41` | `106` categorical part, metadata, and structural tokens (`1` per line) |
| `token_vocabulary_v4.json` | `5f76f8c805065e867ee6242cbc86989c8c3ac37d03c20239b288727007a53193` | Canonical token-to-ID map (`0..105`) shared across released checkpoints |

All Parquet datasets, SQL/BLAST databases, motif registries, and checkpoints are gitignored and hosted in [`hf://buckets/McClain/PlasmidRL`](https://huggingface.co/buckets/McClain/PlasmidRL) and [`hf://buckets/McClain/PlasmidLMEval`](https://huggingface.co/buckets/McClain/PlasmidLMEval).

## Canonical Local Layout

```text
data/
├── special_tokens.txt
├── token_vocabulary_v4.json
├── cluster_v3/                      # Certified sequence-cluster-disjoint split (hf://buckets/McClain/PlasmidRL/datasets/cluster_v3/)
│   ├── train.parquet                # 93,900 pairs (17,863 plasmids, 10,420 clusters)
│   ├── validation.parquet           # 4,728 pairs (981 plasmids, 464 clusters)
│   ├── test.parquet                 # 4,939 pairs (1,023 plasmids, 468 clusters)
│   ├── assignments.parquet          # 103,567 cluster assignments
│   ├── manifest.json                # Split parameters, exclusion counts, partition SHA-256s
│   └── audit/report.json            # Independent BLAST+ v2.17.0 cross-split leakage audit
├── training_pairs_v4.parquet        # Full source corpus (108,468 pairs from 20,644 Addgene plasmids)
├── motif_registry_combined.parquet  # 660-entry BLAST/ORF reward registry
├── plannotate_db.parquet            # 567,340-row pLannotate reference table
├── plannotate_blast_db/             # Compiled BLAST/Diamond/Infernal databases (external download)
└── holdout_full_prompts.parquet     # 22,400 full-prompt holdout pairs (v1.0 source-ID archive)
```

Full byte counts, row counts, and SHA-256 digests are indexed in [`infra/DATA_LOCATIONS.md`](../infra/DATA_LOCATIONS.md).

## Downloading Canonical Datasets

```bash
hf buckets sync hf://buckets/McClain/PlasmidRL/datasets/cluster_v3 ./data/cluster_v3
hf buckets cp hf://buckets/McClain/PlasmidRL/datasets/v4/training_pairs_v4.parquet ./data/training_pairs_v4.parquet
hf buckets cp hf://buckets/McClain/PlasmidRL/datasets/v4/motif_registry_combined.parquet ./data/motif_registry_combined.parquet
hf buckets cp hf://buckets/McClain/PlasmidRL/datasets/v4/plannotate_db.parquet ./data/plannotate_db.parquet
```

## Prompt–Sequence Schema

Each sample in `training_pairs_v4.parquet` and `cluster_v3/*.parquet` formats unordered biological tokens before `<SEQ>` followed by nucleotide sequence:

```text
<BOS><VEC_...><SP_...><COPY_...><AMR_...><ORI_...><PROM_...><ELEM_...><REPORTER_...><TAG_...><SEQ>ATG...TAA<EOS>
```

The dataset loader (`plasmid_llm/data.py`) accepts either `token_prompt`/`token_completion` (with optional `<SEQ>` delimiter) or `prompt`/`sequence`, masks all prompt + delimiter tokens (`-100`) during causal LM cross-entropy training, applies random circular cut-point rotation, and drops `<SP_*>` and `<VEC_*>` context tokens with probability `0.2`.

## Rebuilding the Motif Registry

```bash
PLASMID_ANNOTATION_ROOT=/path/to/addgene_clean python scripts/build_motif_registry.py
```
