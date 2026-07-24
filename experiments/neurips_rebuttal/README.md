# NeurIPS 2026 rebuttal experiments (submission 25202)

Analyses run in response to the reviews. Kept out of `eval/` because these answer specific
reviewer objections rather than forming part of the standing eval suite.

Findings and their write-ups live in [`docs/neurips2026_rebuttal/00_triage.md`](../../docs/neurips2026_rebuttal/00_triage.md).

## Files

| file | what it does | finding | reviewer |
|---|---|---|---|
| `common.py` | one shared detection + scoring path for every panel | — | — |
| `build_benchmark.py` | builds a plasmid-disjoint prompt benchmark from unseen Addgene plasmids | F10 | Dok5 W5/Q1, meta 2 |
| `score.py` | `useful = viable × faithful`, plus precision / unrequested components | F14, F11 | pSXh W2/Q2, meta 1 |
| `baselines.py` | `retrieval` and `completion` nulls | F13 | Dok5 W2, pSXh W4 |
| `similarity.py` | BLAST novelty vs training corpus, with real held-out control | F9 | Dok5 W2, meta 2 |
| `orthogonal.py` | AMRFinderPlus rescoring, independent of the registry | F12 | Dok5 Q3, pSXh Q1 |

## Why one shared scoring path

The meta-review's first objection is that reward, faithfulness metric and best-of-K ranker
all run through the same 660-entry registry and pLannotate install. Adding a second
inconsistency of our own would be worse. So every panel — base, post-GRPO, retrieval,
completion-only, real plasmids — goes through `common.detect` / `common.is_viable`:

```
pLannotate Feature → feature_to_category_token() → percmatch ≥ 95, non-fragment
viable = origin AND selection marker AND (promoter OR CDS)
```

`feature_to_category_token` is the repo's own (`scripts/build_motif_registry.py`), so
"requested" and "detected" always share a vocabulary. The viability rule is left exactly as
the paper defines it — permissive, and Dok5 W3 is right that it means *annotatable* rather
than *would propagate* — because changing it mid-rebuttal would make these numbers
incomparable to the submitted ones. A stricter functional check belongs in a separate,
clearly-labelled metric.

## Order

```bash
# 1. build a benchmark from plasmids never seen in pretraining
python build_benchmark.py --plannotate heldout_ann/plannotate_results.json \
    --metadata metadata.parquet --pool heldout_2000.csv --n 1000 \
    --out disjoint_benchmark.parquet

# 2. generate with each checkpoint (eval/scripts/generate.py), then annotate
#    (eval/scripts/annotate.py), then score
python score.py --benchmark disjoint_benchmark.parquet \
    --panel base=gen_base/plannotate_results.json \
    --panel grpo=gen_grpo/plannotate_results.json \
    --paired base,grpo --precision

# 3. nulls
python baselines.py retrieval --train-pairs training_pairs_v4.parquet \
    --benchmark disjoint_benchmark.parquet --annotations retr_ann/plannotate_results.json
python baselines.py completion --benchmark disjoint_benchmark.parquet \
    --panel shao=pg_base/plannotate_results.json

# 4. independent checks
python similarity.py --train-pairs training_pairs_v4.parquet \
    --generations gen_grpo/generations.parquet --control addgene_500.fasta
python orthogonal.py --amr grpo_amr.tsv --generations gen_grpo/generations.parquet
```

## Inputs not in the repo

| what | where |
|---|---|
| `training_pairs_v4.parquet`, `full_sequences.parquet`, `metadata.parquet` | `s3://phd-research-storage-1758274488/databricks_export/` |
| archived eval runs (generations, annotations, prodigal proteins) | HF bucket `McClain/PlasmidLMEval` |
| `holdout_full_prompts.parquet` | Anyscale bucket — **note: not plasmid-disjoint, see F0b** |
| pLannotate, AMRFinderPlus, blastn | `g6-big`: `plannotate` conda env; `/opt/dlami/nvme/mcclain_analysis/plasmid_llm_analysis/env` |
