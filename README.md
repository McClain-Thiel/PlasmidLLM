# PlasmidLM

Language models for synthetic plasmid DNA sequence generation with reinforcement learning.

## Released Models & Artifacts

- **Base model (`v2.0` cluster-disjoint / `v1.0` source-ID):** [`McClain/PlasmidLM-kmer6`](https://huggingface.co/McClain/PlasmidLM-kmer6)
- **GRPO post-trained (`v2.0` best seed / `v1.0` / all 9 ablation & seed branches):** [`McClain/PlasmidLM-kmer6-GRPO-plannotate`](https://huggingface.co/McClain/PlasmidLM-kmer6-GRPO-plannotate)
- **Evaluation & Reproducibility Archive:** [`hf://buckets/McClain/PlasmidLMEval`](https://huggingface.co/buckets/McClain/PlasmidLMEval) (`rigor_20261116/` & `neurips_2026_rebuttal/`)
- **Datasets & Certified Cluster Split:** [`hf://buckets/McClain/PlasmidRL`](https://huggingface.co/buckets/McClain/PlasmidRL) (`datasets/cluster_v3/` & `datasets/v4/`)
- **Training Runs:** [Weights & Biases (`mcclain/PlasmidLLM`)](https://wandb.ai/mcclain/PlasmidLLM)

Source code, vocabulary files, reference summaries, and unit tests live in Git; model weights, datasets, generated FASTA/Parquets, and annotation archives are hosted on Hugging Face.

## Goal

Train a transformer to generate plasmid DNA sequences that contain specific functional elements — antibiotic resistance genes, origins of replication, promoters, reporters, and other components — specified through a categorical token prompt. The model first learns plasmid structure through pretraining on real plasmid sequences, then is refined with reinforcement learning using sequence alignment rewards to improve motif fidelity.

## Pipeline

```
Pretraining (causal LM)  →  Post-training (RL)  →  Generation (vLLM)  →  Evaluation
```

1. **Pretraining**: Next-token prediction on `<BOS><tokens><SEP>SEQUENCE<EOS>` formatted plasmid data. Learns DNA syntax and the relationship between categorical tokens and sequence content.

2. **Post-training**: Reinforcement learning (GRPO, PPO) with reward functions that use Smith-Waterman alignment to verify the generated sequence actually contains the requested motifs. Runs on Ray with distributed actors.

3. **Generation**: vLLM inference from a token prompt to produce novel plasmid sequences.

4. **Evaluation**: Model quality (does the output contain the right components?) and scorer validity (does the reward function separate good from bad?).

## Repo tour

```
PlasmidLLM/
├── plasmid_llm/              Core library (model, tokenizer, config, data)
│   ├── config.py                 Training config dataclasses
│   ├── data.py                   PyTorch dataset for plasmid sequences
│   └── models/hf_plasmid_lm/    HuggingFace-compatible transformer
│
├── pretraining/              Stage 1 — causal language modeling
│   ├── train.py                  Training script (Trainer-based)
│   └── configs/                  Experiment configs (baseline, k-mer, MoE)
│
├── post_training/            Stage 2 — reinforcement learning on Ray
│   ├── algorithms/               GRPO, PPO implementations
│   ├── scorers/                  Alignment and motif reward functions
│   ├── common/                   ModelActor, loss registry, utilities
│   └── runners/                  End-to-end training scripts
│
├── eval/                     Evaluation framework
│   ├── eval_plannotate.py        End-to-end checkpoint eval (BLAST-based)
│   ├── scorers/                  Scorer validation (separation, AUROC, degradation)
│   ├── scripts/                  Generation, metrics, figures
│   └── plasmidspace/             Visualization UI
│
├── experiments/              Self-contained analyses tied to specific studies
│   └── neurips_rebuttal/         Corrected holdout, registry-blind, and reuse checks
│
├── scripts/                  Utilities (generation, motif registry, HF upload)
├── notebooks/                Analysis notebooks (marimo)
├── data/                     Tracked vocabulary files; large datasets are external
└── tests/                    Unit and integration tests
```

Each stage has its own README with detailed documentation:
- **[pretraining/README.md](pretraining/README.md)** — Model architecture, tokenization, config reference
- **[post_training/README.md](post_training/README.md)** — Actor/algorithm design, scorers, multi-actor training
- **[eval/README.md](eval/README.md)** — Evaluation axes, metrics, scorer pass criteria
- **[data/README.md](data/README.md)** — Vocabulary files and external dataset layout
- **[experiments/neurips_rebuttal/README.md](experiments/neurips_rebuttal/README.md)** — Cluster-disjoint rigor evaluation, rebuttal analyses, and interpretation limits

## Model

Transformer decoder with RoPE, RMSNorm, and optional Mixture-of-Experts. Supports character-level and k-mer tokenization. Fully compatible with HuggingFace `transformers`, `vLLM`, and `trl`.

## Install

```bash
uv sync

# For RL post-training (Ray, parasail, biopython)
uv sync --extra rl

# For the rebuttal analyses
uv sync --extra rebuttal
```

## Quick start

```bash
# Pretrain
python -m pretraining.train pretraining/configs/p4_smoke_test.py

# Post-train with GRPO
python -m post_training.runners.run_grpo

# Generate sequences
python scripts/generate.py \
    --hf-model output/pretraining/final \
    --vocab output/pretraining/final/vocab.json \
    --prompt "<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>"
```

## Dependencies

| Category | Packages |
|----------|----------|
| Core | PyTorch 2.1+, Transformers 4.38+, PyArrow, MLflow |
| RL | Ray 2.9+, parasail 1.3+, Biopython 1.80+ |
| Dev | pytest, ruff |

See `pyproject.toml` for the full list.

Large inputs and outputs are not committed; see [`infra/DATA_LOCATIONS.md`](infra/DATA_LOCATIONS.md) for Hugging Face bucket locations (`hf://buckets/McClain/PlasmidLMEval` and `hf://buckets/McClain/PlasmidRL`).
