# Notebooks

Analysis notebooks for PlasmidLLM evaluation and ablations.

## Files

- `base_vs_grpo_analysis.py` — **marimo** notebook. Compares pre-GRPO (`McClain/PlasmidLM-kmer6`)
  vs post-GRPO (`McClain/PlasmidLM-kmer6-GRPO-plannotate`) generations: bootstrap CIs, Wilson intervals,
  paired McNemar on prompt-level viability, per-category breakdowns.
- `Build_Training_Pairs.py` — legacy marimo/jupytext script for constructing the conditional-prompt
  training set.
- `pretrained_analysis.ipynb` / `rl_MOE_test.ipynb` — older Jupyter notebooks.

## marimo — one-time setup

[marimo](https://marimo.io) is a reactive Python notebook that is stored as a plain `.py` file, so
it diffs cleanly in git and can be reviewed in a PR. It is *not* a Jupyter `.ipynb` with JSON cells.

Install once (into the project `.venv`):

```bash
uv add --dev marimo
```

or, ad-hoc:

```bash
uv run --with marimo marimo edit notebooks/base_vs_grpo_analysis.py
```

## Running a notebook

Interactive edit mode (reads + writes to the `.py` file, supports inline widgets):

```bash
uv run marimo edit notebooks/base_vs_grpo_analysis.py
```

Read-only app mode (for sharing a rendered view, no cell editing):

```bash
uv run marimo run notebooks/base_vs_grpo_analysis.py
```

Headless / script mode (re-executes every cell top-to-bottom as a regular Python file):

```bash
uv run python notebooks/base_vs_grpo_analysis.py
```

## Conventions

- Every notebook lists the HF-bucket run directories and model IDs it loads from at the top, so the
  data provenance is explicit and the notebook is reproducible on any machine with `hf buckets` and
  the relevant read permissions.
- Heavy compute (generation, pLannotate) belongs in `eval/scripts/`, not in notebooks. Notebooks
  consume the pre-computed `metrics/` parquets and `generations.parquet` produced by those scripts.
- Always record the bucket revision (commit hash or upload timestamp) of the run being analyzed in
  the notebook's first cell.
