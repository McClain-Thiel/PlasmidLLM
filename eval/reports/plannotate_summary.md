# pLannotate Evaluation Summary

**Date:** 2026-03-05 | **Machine:** g6-big (NVIDIA L4, 22GB) | **Total runtime:** ~25 min

## Setup

- 50 validation prompts (5% val split, seed=42), 3 candidates each (best-of-3 selection)
- 14 single-token probe tests per model
- Generation: HF `model.generate()` with bf16, batch_size=16, max_tokens=3000 (~9kb DNA)
- MoE model uses dense expert dispatch patch (all experts via bmm, router mask zeroing)
- Annotation: pLannotate batch mode, 8 parallel workers, quality filter (percmatch >= 95%, no fragments)
- Scoring: two-tier token mapping (sseqid registry lookup, then feature name substring match)

## Timing Breakdown

| Phase | Dense (kmer6) | MoE (kmer6-MoE) |
|-------|---------------|------------------|
| Generation (150 seqs) | 6.5 min (924 tok/s) | 6.7 min (851 tok/s) |
| pLannotate (150 seqs, 8 workers) | 3.6 min | 3.0 min |
| Probe generation (14 seqs) | 0.6 min | 0.6 min |
| Probe pLannotate (14 seqs) | 0.4 min | 0.3 min |
| **Subtotal per model** | **~11 min** | **~11 min** |

Model loading ~15s each. Total wall clock including overhead: ~25 min.

## Results

| Metric | Dense (kmer6) | MoE (kmer6-MoE) |
|--------|---------------|------------------|
| Hit Rate (mean) | 48.5% | 46.2% |
| Components Found | 184 / 384 | 179 / 384 |
| Perfect Prompts | 1 / 50 (2%) | 3 / 50 (6%) |
| Avg Sequence Length | 7,373 bp | 7,089 bp |
| EOS Rate | 66% | 64% |
| Generation Speed | 924 tok/s | 851 tok/s |
| Probe Hits | 0 / 14 | 4 / 14 |

## Per-Category Hit Rates

| Category | Dense | MoE |
|----------|-------|-----|
| ORI | 75.0% | 54.2% |
| PROM | 48.8% | 52.0% |
| AMR | 44.8% | 51.7% |
| ELEM | 37.3% | 37.3% |
| REPORTER | 23.5% | 35.3% |
| TAG | 0.0% | 0.0% |

## Probe Test Details (single-token requests)

MoE hit 4/14: ORI_COLE1, ORI_F1, PROM_LAC, ELEM_WPRE. Dense hit 0/14.
Both models generate plasmid-like sequences with recognizable features, but the dense model doesn't reliably place the specific single requested component. Tags (HIS, FLAG) are too short for pLannotate's BLAST detection.

## Key Observations

1. **Models are comparable overall** (~47% hit rate), with MoE slightly better at specific component placement (more perfect prompts, better probes)
2. **ORIs are the strongest category** - well-conserved nucleotide sequences that BLAST finds easily
3. **Tags are invisible to pLannotate** - peptide tags (6xHis, FLAG) are too short for nucleotide BLAST; would need protein-level annotation
4. **Reporters underperform** - likely because they're protein-coding features where nucleotide BLAST is less sensitive than Smith-Waterman protein alignment
5. **~34% of sequences hit max length** (9003 bp) without EOS - longer generation budget would help
6. **pLannotate is the bottleneck** (~1.2s/seq even with 8 workers) - generation is fast at 850-925 tok/s batched
