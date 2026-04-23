# Notebooks

Analysis notebooks for PlasmidLLM evaluation and ablations.

## Files

- `base_vs_grpo_analysis.py` — **marimo** notebook. Compares pre-GRPO (`McClain/PlasmidLM-kmer6`)
  vs post-GRPO (`McClain/PlasmidLM-kmer6-GRPO-plannotate`) generations: bootstrap CIs, Wilson
  intervals, paired McNemar on prompt-level viability, per-category breakdowns, and best-of-$K$
  rejection sampling (the paper's headline figure).
- `soft_token_conditioning.py` — **marimo** notebook. Asks whether the base model conditions on
  the soft / non-reward tags (`BB_`, `COPY_`, `SP_`, `VEC_`) at all, and whether GRPO (which
  never saw them in the reward) perturbed that behavior. Probes each soft value via its
  biologically-expected hard tokens and the existing motif registry; four-bucket classifier
  (correct / wrong / unclassified / no_signal) with Wilson CIs and paired McNemar.

## Opening a notebook — the one command you need

Each notebook declares its dependencies inline (PEP 723 script metadata at the top of the file).
Marimo's `--sandbox` mode hands those deps to `uv`, which builds a throwaway env and launches
the notebook in it. No project-env edits, no global installs:

```bash
uvx marimo edit --sandbox notebooks/base_vs_grpo_analysis.py
```

That's it. `uvx` bootstraps marimo, `--sandbox` reads the PEP 723 header, `uv` caches the env.

## Other modes

Read-only app mode (render a shareable view, no cell editing):

```bash
uvx marimo run --sandbox notebooks/base_vs_grpo_analysis.py
```

Headless / script mode (re-execute every cell top-to-bottom, no browser):

```bash
uv run --script notebooks/base_vs_grpo_analysis.py
```

(The `--script` flag tells `uv` to read the inline PEP 723 header and build an isolated env.)

## About marimo

[marimo](https://marimo.io) is a reactive notebook stored as a plain `.py` file, so it diffs
cleanly in git and can be reviewed in a PR. It is *not* a Jupyter `.ipynb` with JSON cells.
Dependencies live at the top of the file as PEP 723 comments; edit them there, not in
`pyproject.toml`.

## Conventions

- Every notebook lists the HF-bucket run directories and model IDs it loads from at the top, so the
  data provenance is explicit and the notebook is reproducible on any machine with `hf buckets` and
  the relevant read permissions.
- Heavy compute (generation, pLannotate) belongs in `eval/scripts/`, not in notebooks. Notebooks
  consume the pre-computed `metrics/` parquets and `generations.parquet` produced by those scripts.
- Always record the bucket revision (commit hash or upload timestamp) of the run being analyzed in
  the notebook's first cell.
