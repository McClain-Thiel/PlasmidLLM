# Evaluation

Evaluation tools and pipelines for **PlasmidLM**:

1. `scripts/` generates sequences, annotates them with `pLannotate` / `Prodigal` / `AMRFinderPlus` / `dustmasker`, scores external baselines, computes distributional metrics, and regenerates all publication figures.
2. `experiments/neurips_rebuttal/` holds the cluster-disjoint statistical aggregation (`summarize_rigor_evaluation.py`), auxiliary manuscript diagnostics (`manuscript_diagnostics.py`), and historical rebuttal scripts.
3. `scorers/` contains standalone validation notebooks for reward functions (`evaluate_scorer.ipynb`, `plannotate_scorer.ipynb`, `plasmidlike_scorer.ipynb`).
4. `plasmidspace/` is the Gradio interactive sequence explorer.
5. `reference/` holds compact tracked Addgene reference summaries (`addgene_500_3mer_freqs.json`, `addgene_reference_metrics.csv`); generated FASTA/Parquets and rigor archives live under gitignored `eval/runs/` and [`hf://buckets/McClain/PlasmidLMEval`](https://huggingface.co/buckets/McClain/PlasmidLMEval).

## Key Scripts (`eval/scripts/`)

| Script | Purpose |
|---|---|
| `generate.py` | Batched generation across prompt panels and fixed decoding seeds |
| `annotate.py` | Circular `pLannotate v1.2.2`, `Prodigal v2.6.3`, and `dustmasker` annotation runner |
| `score_cluster_v3_baselines.py` | Scores Prompt-Jaccard retrieval, `PlasmidGPT`, `PlasmidGPT-SFT`, and random/dinucleotide controls on `frozen_panel_v3` (Table 2) |
| `build_negative_controls.py` | Generates GC-matched random and dinucleotide-shuffled controls |
| `generate_figures.py` | Deterministically renders all publication figures (Figures 1 & 3–9) from `eval/runs/rigor_20261116/` |
| `compute_metrics.py` | Distributional, codon, and Sourmash similarity metrics |
| `compute_extra_metrics.py` | Extended ORF, per-category hit rates, and motif breakdowns |
| `analyze_best_of_k.py` | Best-of-$K$ reranking analysis helpers |
| `render_report.py` | Renders summary Markdown reports from per-run metric tables |

External bioinformatics dependencies (`pLannotate v1.2.2`, `Prodigal v2.6.3`, `NCBI AMRFinderPlus v4.2.7`, `BLAST+ v2.17.0`) are specified in `config/environment.yml`.
