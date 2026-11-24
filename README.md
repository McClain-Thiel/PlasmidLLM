# PlasmidLM

**PlasmidLM** is a 19.3M-parameter promptable causal DNA language model that translates compositional biological part and metadata tokens into full plasmid sequences in a single autoregressive pass, refined with verifiable sequence-level reinforcement learning (GRPO). Licensed under [Apache-2.0](LICENSE).

## Released Models & Archives

| Artifact | Location | Notes |
|---|---|---|
| **PlasmidLM (supervised base)** | [`McClain/PlasmidLM-kmer6`](https://huggingface.co/McClain/PlasmidLM-kmer6) | `main` (`v2.0`): `100k`-step `cluster_v3` base (`v1` branch preserves `v1.0`) |
| **PlasmidLM (post-GRPO)** | [`McClain/PlasmidLM-kmer6-GRPO-plannotate`](https://huggingface.co/McClain/PlasmidLM-kmer6-GRPO-plannotate) | `main` (`v2.0` seed `44`); `v2-seed{42,43,44}` & ablation branches archived |
| **Evaluation & Rigor Archive** | [`hf://buckets/McClain/PlasmidLMEval`](https://huggingface.co/buckets/McClain/PlasmidLMEval) | `rigor_20261116/` (`frozen_panel_v3` outputs, SHA-256 verified) & `neurips_2026_rebuttal/` |
| **Datasets & Certified Split** | [`hf://buckets/McClain/PlasmidRL`](https://huggingface.co/buckets/McClain/PlasmidRL) | `datasets/cluster_v3/` (`93,900` train / `4,728` val / `4,939` test pairs) & `datasets/v4/` |
| **Training Logs** | [Weights & Biases (`mcclain/PlasmidLLM`)](https://wandb.ai/mcclain/PlasmidLLM) | Pretraining and three-seed GRPO telemetry |

Source code, special-token vocabularies, small reference tables, and unit tests live in this repository; checkpoints, Parquet datasets, generated FASTAs, and annotation outputs live on Hugging Face (indexed in [`infra/DATA_LOCATIONS.md`](infra/DATA_LOCATIONS.md)).

## Quick Start (Inference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "McClain/PlasmidLM-kmer6-GRPO-plannotate"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

prompt = "<BOS><VEC_BACTERIAL><AMR_KANAMYCIN><ORI_COLE1><PROM_T7><SEQ>"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs, do_sample=True, temperature=0.3, top_p=0.95, top_k=0,
    max_new_tokens=4096, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
)
dna = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

## Method Pipeline

1. **Sequence-Cluster-Disjoint Data (`cluster_v3`):** `103,567` prompt–sequence pairs from `19,867` Addgene plasmids partitioned into `11,352` circular clusters (`MMseqs2` + fixed-point `megablast` closure at `>= 90%` identity and `>= 90%` bidirectional coverage; `93,900` train / `4,728` val / `4,939` test pairs; `0` cross-split BLAST hits under `BLAST+ v2.17.0`).
2. **Tokenizer & Architecture:** `4,202` active tokens (`4,208` padded): `4,096` overlapping `6`-mers (stride `3`; `4,096` tokens span `~12.3 kb`) + `106` special tokens (`57` verifiable hard parts, `42` contextual soft metadata tags, `7` structural tokens). `10`-layer, `d=384`, `8`-head, RoPE + RMSNorm causal Transformer (`19.3M` params).
3. **Verifiable GRPO Post-Training:** Three seeds (`42, 43, 44`; `1,000` steps, `G = 8` rollouts, KL `beta = 0.5`, clip `eps = 0.2`) optimizing a prompt-normalized reward combining hard-token `F_1` (`0.6`), complete circular Table-11 `AMR` ORF intactness (`0.3`), and exact translated epitope tag matching (`0.1`) against a compact BLAST/ORF registry (`PlannotateV2Scorer`).
4. **Multi-Axis Evaluation (`frozen_panel_v3`, `n = 468` clusters):** Overlap-resolved `pLannotate v1.2.2` (`percmatch >= 95%`), orthogonal `Prodigal v2.6.3` + `NCBI AMRFinderPlus v4.2.7`, and `dustmasker` (`BLAST+ v2.17.0`). Post-GRPO raises single-shot useful-plasmid yield from `47.7%` to `58.0% +- 0.7%` (`85.5%` Best-of-`4`) and covered bacterial AMR recall from `58.9%` to `65.0%`.

## Repository Tour

```text
PlasmidLLM/
├── plasmid_llm/                  Core library: HF-compatible model, k-mer tokenizer, config, dataset
├── pretraining/                  Cluster-disjoint split builder/auditor, pretraining runner & configs
├── post_training/                Ray ModelActor, GRPO/PPO algorithms, BLAST/ORF scorers, RL configs
├── eval/                         Generation, pLannotate/Prodigal/AMRFinderPlus annotation, baseline scoring, figure generation, PlasmidSpace UI
├── experiments/neurips_rebuttal/ Statistical aggregation (summarize_rigor_evaluation.py), manuscript diagnostics, blind-registry & MOSAIC checks
├── scripts/                      Standalone CLI tools (generate, benchmark_inference, build_motif_registry, upload_to_hf)
├── notebooks/                    Interactive Marimo notebooks (base_vs_grpo_analysis.py, soft_token_conditioning.py)
├── infra/                        Data registry (DATA_LOCATIONS.md) & Anyscale GPU/CPU job runners
├── data/                         Tracked vocabulary files (special_tokens.txt, token_vocabulary_v4.json)
└── tests/                        Unit and integration tests (pytest)
```

Detailed guides: [pretraining](pretraining/README.md) · [post_training](post_training/README.md) · [eval](eval/README.md) · [experiments/neurips_rebuttal](experiments/neurips_rebuttal/README.md) · [data](data/README.md) · [infra/DATA_LOCATIONS.md](infra/DATA_LOCATIONS.md) · [scripts](scripts/README.md)

## Installation & End-to-End Reproduction

```bash
uv sync --extra dev --extra pretrain --extra rl --extra rebuttal
```

External bioinformatics binaries (`blastn`, `makeblastdb`, `dustmasker`, `pLannotate v1.2.2`, `Prodigal v2.6.3`, `NCBI AMRFinderPlus v4.2.7`, `MMseqs2`) are pinned in [`eval/config/environment.yml`](eval/config/environment.yml).

| Stage | Command | Output |
|---|---|---|
| Unit tests & lint | `.venv/bin/pytest -m "not integration" && .venv/bin/ruff check .` | Core logic, tokenizer, reward, split & statistical checks |
| Certified cluster split | `python -m pretraining.build_cluster_split` → `refine_cluster_split` → `verify_cluster_split` | `data/cluster_v3/` (`manifest.json`, `audit/report.json`) |
| Pretraining (`v2.0`) | `python -m pretraining.train pretraining/configs/cluster_disjoint_v2.py` | `checkpoints/...` (`100,000` steps) |
| GRPO post-training | `python -m post_training.runners.run post_training/configs/grpo_reward_ablation.py` | 3-seed RL checkpoints (`42, 43, 44`) |
| Primary & Best-of-`K` stats | `python experiments/neurips_rebuttal/summarize_rigor_evaluation.py --help` | `summary.json`, `all_observations.parquet`, `tables/*.csv` |
| External baselines | `python eval/scripts/score_cluster_v3_baselines.py --help` | `external_baselines_summary.{json,csv}` |
| Auxiliary paper diagnostics | `python experiments/neurips_rebuttal/manuscript_diagnostics.py` | `manuscript_diagnostics.json` (Tables 1/5/6/7/9, virulence screen) |
| Publication figures | `python eval/scripts/generate_figures.py --out-dir eval/figures/paper` | Figures 1 & 3–9 (`.pdf`/`.png`) |

## Citation

```bibtex
@article{thiel2026plasmidlm,
  title   = {{PlasmidLM}: A Promptable {DNA} Language Model via Verifiable-Reward Post-Training},
  author  = {Thiel, McClain and Barnes, Chris P.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.19.725242},
  url     = {https://doi.org/10.64898/2026.05.19.725242},
}
```
