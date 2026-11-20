# NeurIPS 2026 rebuttal analyses

This directory contains the final analysis code used to answer reviewer concerns about
benchmark leakage, reward/evaluator coupling, retrieval, specificity, and sequence reuse.
It intentionally excludes machine-specific launch wrappers, generated outputs, and failed
or superseded experiments. The exact historical runtime bundle is preserved with the
external artifacts.

## Evidence ledger

| question | final evidence | main limitation |
|---|---|---|
| Does the GRPO effect survive a new source set? | Useful rate 30.0% → 39.8% on 1,000 source-plasmid-ID-disjoint prompts; paired delta +9.8 pp, 95% CI [+5.8, +13.8], McNemar p=2e-6 | The sources are not sequence-cluster-disjoint from pretraining |
| Does GRPO recover motifs hidden from its reward? | Exact held-out registry recovery 8.56% → 10.34%; +1.79 pp, 95% CI [+1.16, +2.41], all four decoding seeds positive | Reward and evaluation still use pLAnnotate-style annotations; this is not wet-lab validation |
| Could retrieval solve the submitted metric? | Directly annotated retrieval is 98.6% viable, 95.7% mean faithful, and 98.6% useful | It retrieves existing plasmids and therefore has no novelty |
| Does an independent AMR annotator confirm the gain? | AMRFinderPlus delta +1.0 pp, 95% CI [-1.17, +3.27], p=0.30 | Covers AMR only and is underpowered for the registry-sized effect |
| Does GRPO improve selectivity? | Conditional precision is flat; unrequested components increase from 2.28 to 3.05 per sequence in the blind-registry comparison | pLAnnotate calls are proxies for biological components |
| How much output sequence is reused from training? | At ≥90% identity, median all-subject union coverage is 42.18% base, 60.55% GRPO, and 96.26% for real held-out controls; best-subject medians are 39.38%, 56.85%, and 83.46% | “Less than real controls” is not “no memorization”; GRPO reuses more training DNA than base |

The first benchmark excludes exact pretraining plasmid IDs, not close relatives. An
exhaustive source-to-training check found that, at ≥90% identity, the median best training
subject covers 84.59% of an evaluation source and 37.6% of sources have ≥90% coverage from
one training plasmid. Use **source-plasmid-ID-disjoint**, never “fully disjoint” or
“cluster-disjoint.”

## Files

| file | purpose |
|---|---|
| `common.py` | Shared prompt parsing, pLAnnotate detection, viability, and paired statistics |
| `build_benchmark.py` | Reconstruct prompts from source plasmids whose IDs are absent from pretraining |
| `score.py` | Useful rate, faithfulness, precision, extras, and paired model comparison |
| `score_bestofk.py` | Paper-compatible oracle best-of-K scoring |
| `baselines.py` | Prompt-Jaccard retrieval and unconditional-generation baselines |
| `orthogonal.py` | AMRFinderPlus scoring independent of the reward registry |
| `build_registry_split.py` | Within-token connected-component split of reward and held-out motif references |
| `verify_registry_split.py` | Exhaustive BLAST verification of the registry split |
| `heldout_registry_multiseed.py` | Four-seed exact held-out-reference endpoint |
| `multiseed_generation_guardrails.py` | Paired viability, faithfulness, precision, and extra-component endpoints |
| `exhaustive_training_mosaic.py` | Exhaustive generation/control BLAST against every training plasmid |
| `summarize_exhaustive_blast_thresholds.py` | Identity-threshold sensitivity analysis, including zero-hit queries |

## Reproduction

Install the analysis dependencies and external command-line tools:

```bash
uv sync --extra rebuttal
# Also required on PATH: blastn, makeblastdb, pLAnnotate, Prodigal, AMRFinderPlus.
```

Set `REBUTTAL_ROOT` to the downloaded `neurips_2026_rebuttal` snapshot (`hf://buckets/McClain/PlasmidLMEval/neurips_2026_rebuttal/`):

```bash
export REBUTTAL_ROOT=/path/to/neurips_2026_rebuttal/s3/neurips_rebuttal
```

For the three-seed sequence-cluster-disjoint rigor study (`frozen_panel_v3`), see:
- `pretraining/build_cluster_split.py`, `pretraining/refine_cluster_split.py`, `pretraining/verify_cluster_split.py`
- `pretraining/configs/cluster_disjoint_v2.py`
- `post_training/scorers/plannotate_v2.py` and `post_training/configs/grpo_reward_ablation.py`
- `experiments/neurips_rebuttal/summarize_rigor_evaluation.py` and `experiments/neurips_rebuttal/orthogonal.py`
- Archived outputs: `hf://buckets/McClain/PlasmidLMEval/rigor_20261116/`

The canonical blind-registry inputs and results are under
`$REBUTTAL_ROOT/m13_blind_v2_g7e/`; checkpoints are under the sibling
`$REBUTTAL_ROOT/grpo_m13_blind_v2_g7e_ckpt/`. The exhaustive BLAST inputs, raw TSVs,
summaries, and checksums are under
`$REBUTTAL_ROOT/exhaustive_training_mosaic_20260727/`.

For the original source-ID-disjoint comparison:

```bash
uv run --extra rebuttal python experiments/neurips_rebuttal/score.py \
  --benchmark "$REBUTTAL_ROOT/artifacts/disjoint_benchmark.parquet" \
  --panel "base=$REBUTTAL_ROOT/artifacts/gen_base/plannotate_results.json" \
  --panel "grpo=$REBUTTAL_ROOT/artifacts/gen_grpo/plannotate_results.json" \
  --paired base,grpo --precision
```

For the blind-registry endpoint, pass matched seeds 42–45 from the archived base and
treatment annotation directories to `heldout_registry_multiseed.py`; the script rejects
missing or unmatched seeds. For exhaustive sequence reuse, run
`exhaustive_training_mosaic.py` first and then
`summarize_exhaustive_blast_thresholds.py`, explicitly adding any zero-length/zero-hit
queries reported by generation.

## Superseded analyses

- Any BLAST analysis using `-max_target_seqs 5` is invalid for corpus-wide mosaic
  coverage and must not be cited.
- The initial novel-combination “0% full specification” result was confounded by component
  producibility. In the producible subset, novel-context component recovery is roughly
  half of matched familiar-context recovery, not zero.
- Prompt-conditional lift distinguishes prompted from unconditional generation, but a
  prompt-matched retrieval baseline has still higher lift. It is a diagnostic, not
  evidence of superiority over retrieval.
- Failed CFG and targeted-GRPO repair attempts are retained in the external historical
  archive for provenance, not in this release directory.
