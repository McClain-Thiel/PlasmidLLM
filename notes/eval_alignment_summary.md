# Alignment-Based Evaluation Summary

**Date:** 2026-03-06 | **Machine:** g6-big (NVIDIA L4, 22GB) | **Total runtime:** ~2.5 hours

## Setup

- 50 validation prompts (5% val split, seed=42), best-of-3 selection
- 7 temperatures: 0.0 (greedy), 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
- Generation: HF `model.generate()` with bf16, batch_size=16, max_tokens=3000 (~9kb DNA)
- Scoring: parasail SIMD Smith-Waterman against `motif_registry_combined.parquet` (588 entries, 57 tokens)
  - DNA: both strands, gap open=5, extend=1
  - Protein: 6-frame translation for CDS features (AMR, REPORTER, TAG), BLOSUM62, gap open=10, extend=1
  - Cutoff: 70% normalized score ratio (score / self-alignment score)
- FP scan enabled: all 57 tokens checked per sequence
- MoE model uses dense expert dispatch patch (bmm + router mask)

## Temperature Sweep Results

### Dense (PlasmidLM-kmer6)

| Temp | Hit Rate | Precision | TP | FN | FP | EOS% | Avg Len | TPS |
|------|----------|-----------|-----|-----|-----|------|---------|-----|
| **0.0** | 38.2% | 26.3% | 146 | 228 | 298 | 65% | 6714 | 944 |
| 0.1 | 60.8% | 41.6% | 237 | 137 | 328 | 63% | 7210 | 871 |
| **0.3** | **62.4%** | 40.1% | **245** | **139** | 345 | 56% | 7678 | 907 |
| 0.5 | 53.5% | 37.8% | 202 | 182 | 345 | 64% | 7351 | 884 |
| 0.7 | 52.4% | 37.2% | 200 | 184 | 342 | 62% | 7611 | 930 |
| 0.9 | 45.8% | 39.3% | 177 | 207 | 250 | 60% | 7180 | 933 |
| 1.0 | 47.3% | 41.3% | 173 | 211 | 259 | 62% | 7303 | 876 |

**Best: temp=0.3** (62.4% hit rate, 245/384 components found)

### MoE (PlasmidLM-kmer6-MoE)

| Temp | Hit Rate | Precision | TP | FN | FP | EOS% | Avg Len | TPS |
|------|----------|-----------|-----|-----|-----|------|---------|-----|
| **0.0** | 34.0% | 33.5% | 135 | 249 | 200 | 70% | 6479 | 833 |
| 0.1 | 41.9% | 34.9% | 174 | 210 | 241 | 66% | 6837 | 816 |
| 0.3 | 48.2% | 35.1% | 187 | 197 | 245 | 54% | 7322 | 839 |
| 0.5 | 46.4% | 42.4% | 187 | 197 | 247 | 54% | 7351 | 862 |
| **0.7** | **50.6%** | 44.3% | **201** | **183** | 229 | 58% | 7240 | 857 |
| 0.9 | 49.8% | 40.5% | 203 | 181 | 238 | 70% | 6770 | 791 |
| 1.0 | 42.4% | 43.0% | 172 | 212 | 213 | 64% | 6770 | 777 |

**Best: temp=0.7** (50.6% hit rate, 201/384 components found)

## Per-Category Breakdown (best temps)

| Category | Dense (0.3) | | MoE (0.7) | |
|----------|-------------|------------|-----------|-----------|
| | Hit Rate | FP | Hit Rate | FP |
| ORI | **80.6%** | 34 | 61.1% | 19 |
| AMR | **67.2%** | 34 | 53.4% | 24 |
| PROM | **66.9%** | 72 | 53.5% | 47 |
| ELEM | **55.9%** | 82 | 44.1% | 49 |
| REPORTER | 29.4% | 84 | **64.7%** | 62 |
| TAG | 12.5% | 39 | **25.0%** | 28 |

## Key Observations

1. **Dense model is better overall** (62.4% vs 50.6% best hit rate), especially for ORI, AMR, PROM, and ELEM categories.

2. **MoE is better at reporters and tags** — protein-coding features where the MoE's routing may help with codon-level patterns.

3. **Greedy decoding is worst for both models** (38.2% and 34.0%). Some temperature diversity is critical — the models need randomness to explore the sequence space. Best-of-3 selection can't compensate for greedy's repetitive outputs.

4. **Optimal temperatures differ**: Dense peaks at 0.3 (low), MoE peaks at 0.7 (moderate). The MoE may need more randomness to overcome routing noise.

5. **False positives are abundant** (~250-345 per model). Both models generate biologically realistic plasmids with many recognizable features beyond what's requested. Common FPs include REPORTER_EGFP/GFP/YFP/mEmerald (fluorescent proteins are similar), ORI_F1, PROM_LAC — features that commonly co-occur in real plasmids.

6. **Precision is low** (~35-44%) because of the high FP rate. This isn't necessarily bad — it means the models generate complete, realistic plasmids rather than minimal constructs.

7. **FP decreases at higher temps** for both models (345→259 for dense, 245→213 for MoE), suggesting high temperature produces less structured sequences with fewer recognizable features.

8. **Reporter detection improved vs pLannotate eval** — Smith-Waterman protein alignment catches reporters (64.7% MoE) that pLannotate's nucleotide BLAST missed (35.3%).

## Comparison with pLannotate Evaluation

| Metric | pLannotate (temp=0.7) | SW Alignment (best temp) |
|--------|----------------------|--------------------------|
| Dense hit rate | 48.5% | 62.4% (temp=0.3) |
| MoE hit rate | 46.2% | 50.6% (temp=0.7) |
| Dense ORIs | 75.0% | 80.6% |
| MoE reporters | 35.3% | 64.7% |
| FP detection | No | Yes (avg ~300/model) |

Smith-Waterman alignment is more sensitive, especially for protein-coding features, and allows temperature optimization.
