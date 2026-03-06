# PlasmidLM

Language models for synthetic plasmid DNA sequence generation with reinforcement learning.

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
│   ├── models/                   Model quality metrics (recall, precision, per-component)
│   ├── scorers/                  Scorer validation (separation, AUROC, degradation)
│   └── plasmidspace/             Visualization UI (TBD)
│
├── evaluation/               Evaluation scripts (alignment, plannotate, checkpoint)
├── scripts/                  Utilities (generation, motif registry, HF upload)
├── notebooks/                Analysis notebooks
├── data/                     Sample data + vocabularies (see data/README.md)
├── tests/                    Unit and integration tests
└── notes/                    Working notes
```

Each stage has its own README with detailed documentation:
- **[pretraining/README.md](pretraining/README.md)** — Model architecture, tokenization, config reference
- **[post_training/README.md](post_training/README.md)** — Actor/algorithm design, scorers, multi-actor training
- **[eval/README.md](eval/README.md)** — Evaluation axes, metrics, scorer pass criteria
- **[data/README.md](data/README.md)** — Data formats, file inventory, known issues

## Model

Transformer decoder with RoPE, RMSNorm, and optional Mixture-of-Experts. Supports character-level and k-mer tokenization. Fully compatible with HuggingFace `transformers`, `vLLM`, and `trl`.

## Install

```bash
pip install -e .

# For RL post-training (Ray, parasail, biopython)
pip install -e ".[rl]"
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
