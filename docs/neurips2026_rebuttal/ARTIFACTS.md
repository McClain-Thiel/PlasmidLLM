# Rebuttal artifact index — submission 25202

Everything produced for the NeurIPS 2026 rebuttal, where it lives durably, and what produced
it. **The session scratchpad (`/private/tmp/...`) and g6-big's NVMe (`/opt/dlami/nvme/`) are
both ephemeral — nothing here depends on them.** All artifacts are mirrored to S3.

- **Findings & analysis:** [`00_triage.md`](00_triage.md) (git) — 16 findings F0–F16
- **Scripts:** [`experiments/neurips_rebuttal/`](../../experiments/neurips_rebuttal/) (git)
- **Research log:** Notion Research Log, entry dated 2026-07-24 (id `3a7d0098-937c-81ed-beab-d05367dc27fd`)
- **Durable data root:** `s3://phd-research-storage-1758274488/neurips_rebuttal/`
- **Branch:** `claude/neurips-rebuttal-triage-59f713` (pushed to GitHub)

## S3 layout

```
s3://phd-research-storage-1758274488/neurips_rebuttal/
├── artifacts/          full mirror of g6-big:/opt/dlami/nvme/rebuttal/ (169 MB)
│   ├── gen_base/  gen_grpo/                disjoint-benchmark generations + annotations (F14)
│   ├── bok_{base,grpo}_s{43,44,45}/        best-of-4 extra seeds + annotations (F15)
│   ├── nov_base/  nov_grpo/                novel-combination generations + annotations (F16)
│   ├── retr_ann/                           retrieved-plasmid annotations (F13)
│   ├── heldout_ann/                        pLannotate of the 2,000 held-out plasmids (F10)
│   ├── comp_shao/  comp_cunn/              completion-only generations (in progress)
│   ├── orth/                               AMRFinderPlus tsv + proteins (F12)
│   └── scripts/                            copies of the generate/annotate scripts used
└── derived/            benchmarks + intermediate tables built off-box
    ├── disjoint_benchmark.parquet          1,000 unseen-plasmid prompts (F10/F14)
    ├── novel_benchmark.parquet             500 novel-combination prompts (F16)
    ├── heldout_2000.{csv,fasta}            the sampled unseen plasmids
    ├── retrieved_ids.txt                   retrieval picks per prompt (F13)
    └── {gen,real}_vs_train.tsv             BLAST similarity tables (F9)
```

Inputs (pre-existing, not re-uploaded): `databricks_export/{training_pairs_v4,full_sequences,metadata,annotations}.parquet`; HF bucket `McClain/PlasmidLMEval` (the paper's original runs); Anyscale bucket `holdout_full_prompts.parquet` (**note: not plasmid-disjoint, F0b**).

## Finding → script → data

| finding | one-line result | script | key data |
|---|---|---|---|
| F9 | outputs 43% training-explained vs 81% real | `similarity.py` | `derived/{gen,real}_vs_train.tsv` |
| F10 | disjoint benchmark constructed (120k unseen) | `build_benchmark.py` | `derived/disjoint_benchmark.parquet`, `artifacts/heldout_ann/` |
| F11 | GRPO adds unrequested components | `score.py --precision` | `artifacts/{gen_base,gen_grpo}/` |
| F12 | AMRFinderPlus: gain not significant | `orthogonal.py` | `artifacts/orth/` |
| F13 | retrieval saturates useful rate at 95% | `baselines.py retrieval` | `derived/retrieved_ids.txt`, `artifacts/retr_ann/` |
| F14 | **GRPO +9.8pp on unseen plasmids, p=2e-6** | `score.py --paired` | `artifacts/{gen_base,gen_grpo}/` |
| F15 | best-of-4 67.0% → 80.5% | `score_bestofk.py` | `artifacts/bok_*/` |
| F16 | **model does not compose (0% full-spec)** | `novel_combination.py`, `score_novel.py` | `derived/novel_benchmark.parquet`, `artifacts/nov_*/` |
| F17 | **unconditional gen matches useful rate** | `generate_completion.py`, `baselines.py completion` | `artifacts/comp_*/`, `derived/comp_*_plannotate_results.json` |
| F18 | **prompt-conditional lift: real control, +7.8pp vs ~0 for baselines** | `control_lift.py` | `artifacts/{gen_grpo,comp_shao,comp_cunn}/` |
| CFG fix | decode-time steering fails | `cfg_generate.py` | `artifacts/cfg_smoke/` |
| GRPO fix | novel-combo rollouts (running) | `build_rollout_prompts.py`, `grpo_novel_aug_config.py` | `derived/rollout_novel_aug.parquet`, `artifacts/grpo_novel_ckpt/` |

| F19 | GRPO-on-novel-combos fix failed | `grpo_novel_aug_config.py` | `artifacts/grpo_novel_ckpt/`, `fix_{novel,disjoint}/` |
| F20 | F16 confounded; composition ~49% for producible components | (analysis) | `derived/novel_benchmark.parquet` |
| F21 | producible-targeted GRPO: reward is the bottleneck | `grpo_producible.py`, `control_lift.py` | `derived/{novel_producible_benchmark,rollout_producible}.parquet`, `derived/{fix100,orig_prod}_plann.json`, `artifacts/grpo_producible_ckpt/` |

Setup for the F16 fix runs and their torch-2.9 env patches: [`F16_FIX_SETUP.md`](F16_FIX_SETUP.md).

## Exact commands to reproduce the headline numbers

```bash
# pull data
aws s3 sync s3://phd-research-storage-1758274488/neurips_rebuttal/ ./rebuttal_data/
cd experiments/neurips_rebuttal

# F14 — the keystone
python score.py --benchmark ../../rebuttal_data/derived/disjoint_benchmark.parquet \
    --panel base=../../rebuttal_data/artifacts/gen_base/plannotate_results.json \
    --panel grpo=../../rebuttal_data/artifacts/gen_grpo/plannotate_results.json \
    --paired base,grpo --precision

# F15 — best-of-4
python score_bestofk.py --benchmark ../../rebuttal_data/derived/disjoint_benchmark.parquet \
    --draws base=<dj_base>,<bok_base_s43>,<bok_base_s44>,<bok_base_s45> \
    --draws grpo=<dj_grpo>,<bok_grpo_s43>,<bok_grpo_s44>,<bok_grpo_s45>

# F16 — novel combinations
python score_novel.py --benchmark ../../rebuttal_data/derived/novel_benchmark.parquet \
    --train-pairs <training_pairs_v4.parquet> \
    --panel base=../../rebuttal_data/artifacts/nov_base/plannotate_results.json \
    --panel grpo=../../rebuttal_data/artifacts/nov_grpo/plannotate_results.json
```

## Models

| role | checkpoint |
|---|---|
| PlasmidLM base (released) | `McClain/PlasmidLM-kmer6` (SHA256 `a75f5821…` = on-disk `plasmid_lm_kmer6_local`) |
| PlasmidLM post-GRPO | `McClain/PlasmidLM-kmer6-GRPO-plannotate` |
| PlasmidGPT / Shao 2024 | `UCL-CSSB/PlasmidGPT` (ppl 20.0, healthy) |
| PlasmidGPT-SFT / Cunningham | `UCL-CSSB/PlasmidGPT-SFT` (ppl 13.5, healthy) |
| PlasmidGPT-GRPO / ICML | `UCL-CSSB/PlasmidGPT-GRPO` (**ppl 1337 — degraded, set aside**) |

## Compute

`g6-big` (i-0cda00ffb3cacfc12, g6.4xlarge, L4 23 GB), `ssh g6-big` → 52.54.249.131.
Envs: `plannotate` conda (pLannotate 1.2.2, blastn, prodigal); AMRFinderPlus 4.2.5 + HMMER 3.4
at `/opt/dlami/nvme/mcclain_analysis/plasmid_llm_analysis/env`; `/opt/dlami/nvme/plasmid_venv`
(torch 2.9.1+cu128, transformers 4.57.6) for generation.
