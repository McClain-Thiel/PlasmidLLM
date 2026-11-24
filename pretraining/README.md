# pretraining

Causal language modeling and sequence-cluster-disjoint data partitioning for **PlasmidLM**.

## Quick Start

```bash
# 1. Smoke test (100 steps, verifies pipeline end-to-end)
python -m pretraining.train pretraining/configs/p4_smoke_test.py

# 2. Build and audit the sequence-cluster-disjoint split (cluster_v3)
python -m pretraining.build_cluster_split --help
python -m pretraining.refine_cluster_split --help
python -m pretraining.verify_cluster_split --help

# 3. Pretrain the released v2.0 base checkpoint (McClain/PlasmidLM-kmer6)
PLASMID_SPLIT_DIR=data/cluster_v3 \
PLASMID_SPLIT_AUDIT=data/cluster_v3/audit/report.json \
PLASMID_RUN_ID=cluster_base \
  python -m pretraining.train pretraining/configs/cluster_disjoint_v2.py
```

## Certified Sequence-Cluster-Disjoint Split

Source-plasmid-ID splitting alone leaves near-identical vector backbones across train and test splits. The clustering pipeline eliminates backbone leakage:

1. `build_cluster_split.py` filters records containing non-`ACGTN` ambiguity codes (`777` plasmids / `4,901` pairs), canonicalizes circular DNA sequences, runs `MMseqs2` at `>= 90%` identity and `>= 90%` bidirectional coverage, and assigns connected components to `train` / `validation` (`5%`) / `test` (`5%`).
2. `refine_cluster_split.py` runs fixed-point `megablast` closure over cross-split edges (`3` iterations) to merge any missed circularly equivalent backbones into `11,352` disjoint clusters (`93,900` train / `4,728` val / `4,939` test pairs).
3. `verify_cluster_split.py` runs an independent `BLAST+ v2.17.0` cross-split audit (`0` hits allowed at `>= 90%` identity and `>= 90%` circular coverage) and writes `audit/report.json` with exact partition SHA-256 digests enforced at training startup by `cluster_disjoint_v2.py`.

## Sequence Format, Tokenization, and Augmentation

Each row is formatted with unordered special tokens before `<SEQ>` followed by DNA sequence:

```text
<BOS><VEC_...><SP_...><COPY_...><AMR_...><ORI_...><PROM_...>...<SEQ>ATG...TAA<EOS>
```

- **Vocabulary (`4,202` active / `4,208` padded):** `4,096` overlapping `6`-mers with stride `3` (`3` bp per token; `4,096`-token context spans `~12.3 kb`) plus `106` special tokens (`57` verifiable hard parts, `42` contextual soft metadata tags, `7` structural tokens).
- **Architecture:** `10`-layer Llama-style causal Transformer (`d = 384`, `8` heads, `1,536` GELU FFN, RMSNorm `1e-5`, RoPE `theta = 10,000`, tied embeddings, `19,318,656` parameters).
- **Pretraining regime (`cluster_disjoint_v2.py`):** Cross-entropy loss over DNA + `<EOS>` (prompt tokens masked at `-100`) for `100,000` steps (`effective batch = 64`, AdamW `lr = 3e-4`, cosine schedule with `1,000` warmup steps, weight decay `0.1`, `bfloat16`), random circular cut-point rotation, independent `0.2` dropout on `<SP_*>` / `<VEC_*>` context tags, and inverse-square-root minority oversampling (target fraction `0.1`, cap `4.0x`) across `<ORI_P15A>`, `<ORI_PSC101>`, `<AMR_CHLORAMPHENICOL>`, and `<COPY_LOW>`.

## Configs

| Config | Purpose | Status |
|---|---|---|
| `configs/cluster_disjoint_v2.py` | Released `v2.0` `100k`-step `cluster_v3` base model (`McClain/PlasmidLM-kmer6`) | **Current primary** |
| `configs/p4_kmer6_s3.py` | Archived `v1.0` `65k`-step `6`-mer stride-`3` base on `training_pairs_v4.parquet` | Legacy `v1.0` (`v1` branch) |
| `configs/p4_smoke_test.py` | `100`-step end-to-end smoke test | Smoke test |
| `configs/pretrain_g6big.py` | Single-GPU dense baseline (`char` or `kmer`) | Experimental |
