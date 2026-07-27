# Evaluation

Evaluation has three layers:

1. `scripts/` generates sequences, annotates them, computes metrics, and renders figures.
2. `scorers/` contains exploratory scorer-validation notebooks.
3. `reports/` preserves the March dense-versus-MoE alignment evaluation and its
   machine-readable result.

`plasmidspace/` is the interactive sequence explorer. `reference/` contains the tracked
Addgene control panel and compact reference statistics used by the standing eval suite.
Generated sequences, annotations, model checkpoints, and large training datasets are
external artifacts and are intentionally ignored by Git.

## Current versus historical results

The reports under `reports/` document the March 2026 model-selection evaluation. They are
not the source for the later NeurIPS rebuttal claims. In particular, their original
memorization proxy and their prompt split do not establish plasmid-level independence.

The corrected source-ID holdout, retrieval, held-out reward-registry, specificity,
orthogonal-AMR, and exhaustive BLAST analyses live in
[`experiments/neurips_rebuttal/`](../experiments/neurips_rebuttal/). That README is the
current evidence ledger and records which earlier interpretations are superseded.

## Typical pipeline

```bash
# Generate a panel.
python eval/scripts/generate.py --help

# Annotate it with pLAnnotate/Prodigal.
python eval/scripts/annotate.py --help

# Compute distributional and biological metrics.
python eval/scripts/compute_metrics.py --help
```

External tools and databases are pinned in `config/environment.yml`. Small reference
tables remain in Git; run directories belong under ignored `eval/runs/` or in the
external artifact store.
