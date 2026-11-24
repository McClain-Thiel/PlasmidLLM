# Rigor Evaluation & Rebuttal Analyses

This directory contains the evaluation, statistical inference, orthogonal validation, and sequence-reuse scripts for **PlasmidLM**.

## 1. Sequence-Cluster-Disjoint Three-Seed Study (`cluster_v3` / `rigor_20261116`, Paper Primary Benchmark)

Replaces source-plasmid-ID splits with an independently audited circular `MMseqs2` + fixed-point `megablast` split (`11,352` clusters; `0` cross-split BLAST hits at `>= 90%` identity and `>= 90%` bidirectional coverage) and compares `PlasmidLM (supervised base)` (`cluster_base_s42`) against three independent GRPO-v2 post-training seeds (`exp2_v2_s42..44`) on `n = 468` held-out clusters (`frozen_panel_v3`) across `4` decoding seeds (`42–45`):

- **Primary yield (`T = 0.5`):** useful-plasmid rate rises from `47.7%` to `58.0% +- 0.7%` (`+10.3 pp`, `p_Holm = 0.0012`), reaching `74.7% +- 0.9%` at Best-of-`2` and `85.5% +- 0.8%` at Best-of-`4` (`99.2% +- 0.4%` viable).
- **Orthogonal protein validation (`Prodigal v2.6.3` + `NCBI AMRFinderPlus v4.2.7`):** intact bacterial resistance-gene recovery increases from `55.2%` to `61.0% +- 0.2%` across all `11` classes (`p_Holm = 0.0012`) and `58.9%` to `65.0% +- 0.3%` across the `9` bacterial classes covered by AMRFinderPlus (`82.7%` at Best-of-`4`); `0` virulence hits (`--plus`) across `7,488` evaluated generations.
- **Unseen combinations & sequence novelty:** consistent gains on both seen (`n = 360`: `48.7% -> 58.9%`) and unseen (`n = 108`: `44.4% -> 54.9%`, `p = 0.0004`) hard-token specifications; exhaustive `megablast` against all `18,895` unique training plasmids confirms `81.1% +- 0.9%` novelty (`< 90%` bidirectional coverage) for post-GRPO vs. `72.2%` for base and `0.0%` for retrieval.
- **Clean-prompt sensitivity & failure modes:** masking noisy upstream `<ELEM_IRES>`/`<ELEM_TRACRRNA>` prompt labels raises the held-out Addgene `Mean F` ceiling from `75.8%` to `100.0%` and post-GRPO exact `T = 1.00` usefulness from `17.0%` to `23.6%` (`p = 0.0002`); rare origins (`<ORI_P15A>` `7.4%`, `<ORI_PSC101>` `13.1%`), `<COPY_LOW>` (`1.9%` compatible alone), peptide tags (`+1.8 pp`, `p = 0.4452`), and `4,096`-token length drift (`70.8%` cap hits) remain documented bottlenecks.

### Key Scripts (`cluster_v3`)

| File | Purpose |
|---|---|
| `summarize_rigor_evaluation.py` | Main summary, Best-of-`K`, paired `10,000`-resample cluster+seed bootstrap CIs, Holm–Bonferroni tests, CSV tables, and `REPORT.md` |
| `manuscript_diagnostics.py` | Reproduces clean-prompt sensitivity (Table 6), complete BoK-2/4 orthogonal metrics (Tables 1 & 5), Table 7 AMR-11 SDs, `<EOS>` stratification (Table 9A), Seen/Unseen split (Table 9B), virulence screen (Sec. 6), and exhaustive `megablast` novelty (Table 9C) |
| `orthogonal.py` | `Prodigal` + `NCBI AMRFinderPlus` intact resistance-gene scoring |
| `common.py` | Shared token parsing, `pLannotate` detection, viability rules, and paired bootstrap statistics |

Archived outputs: [`hf://buckets/McClain/PlasmidLMEval/rigor_20261116/`](https://huggingface.co/buckets/McClain/PlasmidLMEval) (verified by `eval/runs/rigor_20261116/SHA256SUMS`).

```bash
uv run --extra rebuttal python experiments/neurips_rebuttal/manuscript_diagnostics.py
```

---

## 2. Historical `v1.0` Source-ID-Disjoint & Blind-Registry Archive (`neurips_2026_rebuttal`)

Preserved for provenance alongside `hf://buckets/McClain/PlasmidLMEval/neurips_2026_rebuttal/`. Because source-plasmid-ID splitting (`v1.0`) leaves near-identical backbones across splits (`84.59%` median best training-subject coverage), all primary claims in the paper supersede this earlier split with the certified `cluster_v3` partition above.

| File | Purpose |
|---|---|
| `build_benchmark.py` | Source-plasmid-ID-disjoint prompt reconstruction (`v1.0`) |
| `score.py`, `score_bestofk.py`, `baselines.py` | Source-ID-disjoint scoring and retrieval baselines |
| `build_registry_split.py`, `verify_registry_split.py`, `heldout_registry_multiseed.py` | Within-token blind-registry split (`8.56% -> 10.34%`) |
| `multiseed_generation_guardrails.py` | Four-seed paired guardrail checks |
| `exhaustive_training_mosaic.py`, `summarize_exhaustive_blast_thresholds.py` | Exhaustive BLAST sequence reuse on `v1.0` |
| `reward_preflight.py` | Pre-run reward component sanity checks |

## Methodology Notes

- Always use full-corpus `-max_target_seqs |train|` for exhaustive `megablast` mosaic coverage (`-max_target_seqs 5` is invalid).
- Failed historical CFG and targeted-GRPO pilots remain in external provenance storage only.
