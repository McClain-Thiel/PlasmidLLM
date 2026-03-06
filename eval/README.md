# eval

Evaluation for PlasmidLLM. Two independent axes — **model quality** and **scorer validity** — plus a visualization layer.

```
eval/
├── data/               # Fixture parquets (tracked in git)
│   └── training_pairs_sample.parquet
├── models/             # Does the model produce good plasmids?
├── scorers/            # Do the scorers separate good from bad?
│   └── evaluate_scorer.ipynb
└── plasmidspace/       # UI for exploring generated plasmids (TBD)
```

## data/

Shared fixtures as parquet files. Uses `training_pairs_sample.parquet` from the top-level `data/` directory (same file used by integration tests). Negative classes (random DNA, shuffled, mismatched, truncated) are derived programmatically — no separate files needed.

## models/

Evaluates model output quality: given a prompt requesting specific plasmid components, annotate the generated sequence and measure TP/FN/FP, per-component alignment quality (pct_id, coverage, norm_score), and aggregate metrics across checkpoints and temperatures.

## scorers/

Validates that scorers produce a useful RL training signal by measuring separation between real plasmids and negatives. See `evaluate_scorer.ipynb`.

## plasmidspace/

TBD. Interactive explorer for generated plasmids.
