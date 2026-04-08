# Eval Suite Log

## 2026-04-08

**11:30** — Phase 0 started. Installed sourmash 4.3, minimap2 2.30, lightgbm 4.6, shap 0.49, sklearn 1.7 into plannotate conda env. Fixed sourmash pkg_resources issue (setuptools<70).

**11:45** — Downloaded Addgene-500 reference panel from HF bucket `McClain/PlasmidRL/reference/`. 500 plasmids, 418 not in training set. Pre-computed: length, GC, longest ORF, MFE density, 3-mer freqs.

**11:50** — Pinned tool versions in `eval/config/environment.yml`. pLannotate 1.2.2 (snapgene DB 2021-11), prodigal 2.6.3, BLAST+ 2.17.0.

**11:55** — Built `eval/config/feature_categories.yaml` mapping pLannotate feature types to Tier 3 booleans (origin, selection_marker, promoter, terminator, cds).

**12:00** — Built training-set sourmash signature: 20,644 unique plasmids, k=31 scaled=100. Cached at `eval/reference/training_sigs.zip`.

**12:05** — Building negative controls (random GC-matched, dinucleotide-shuffled, real held-out). Writing generation + annotation pipeline scripts.

**12:30** — Negative controls built (500 each: random, shuffled, real). Wrote pipeline scripts: generate.py, annotate.py, compute_metrics.py, firstlight_sweep.py.

**13:00** — Tested generation: 5 seqs in 21s, mean 8.1kb, 54% GC. Fixed tokenizer padding issue (custom PlasmidKmerTokenizer doesn't expose pad/eos tokens properly).

**14:00** — Tested annotation pipeline end-to-end: pLannotate 3.1s/seq, prodigal 0.3s total, dustmasker 0.1s total. Results look good — 3/5 test sequences show >60% pLannotate coverage.

**14:15** — Launched Phase 1 first-light sweep: 12 cells (temp×top_p×rep_pen), 50 seqs/cell = 600 total. Model loaded once, reused across cells. ETA ~45-60 min.

**14:16** — huggingface-hub version conflict flagged: upgraded to 1.9.2 for bucket API, but transformers needs <1.0. Downgraded to 0.36.2. Bucket uploads will need separate handling.

**14:50** — First-light sweep complete. 12 cells × 50 seqs = 600 generations in 29.6 min. pLannotate annotation initially failed due to `--no-banner` flag — ran retroactively on all cells (~3 min total).

**14:55** — **Winner: t0.7_p0.95_r1.0** (temp=0.7, top_p=0.95, rep_pen=1.0). 70.8% median pLannotate coverage, 22.1 mean features, 7.6kb mean length. Clear winner — temp=0.7 dominates, higher temps drop coverage significantly. Repetition penalty doesn't help.

**15:05** — Extended sweep to test temp=0.3 and 0.5. Coverage plateaus at ~70% across 0.3-0.7, but temp=0.7 has longest sequences (7.6kb, closest to Addgene ref 7.5kb) and fewest <1kb failures. Keeping temp=0.7.

**15:15** — Uploaded all Phase 0+1 data to HF bucket `McClain/PlasmidLMEval` (81 files, 17.2 MB). Used `hf buckets sync` CLI — the Python `batch_bucket_files()` API silently fails for non-README files.

**15:20** — Phase 2 started: generating 1000 plasmids with winning config (temp=0.7, top_p=0.95, top_k=50, no rep penalty). Model: `McClain/PlasmidLM-kmer6-GRPO-plannotate`.

**16:06** — Phase 2 complete. 1000 sequences in 46.3 min (21.6 seq/min). Mean length 7.2kb, median 9.0kb, 50.6% GC, 48.7% EOS rate, 5.6% <1kb. Baselines also annotated: random 5.2% hit rate, shuffled 7.6%, real 100%.

**16:07** — Phase 3 started: running pLannotate (8 workers) + Prodigal + dustmasker on all 1000 generated sequences.

**16:40** — Phase 3 complete. pLannotate: ~2s/seq with 8 workers on 1000 seqs. Prodigal and dustmasker fast (<30s total).

**16:41** — Phase 4 metrics computed on all 1000 generated sequences:

### Tier 1: Distributional
- Length KS stat: 0.256 (p < 1e-19) — significant difference from reference (generated skews longer)
- GC KS stat: 0.121 (p = 0.0001) — slight difference, generated slightly lower GC
- Wasserstein distances: length=1338bp, GC=0.020

### Tier 3: Essentials
| Metric | Generated (n=1000) | Real Addgene (n=500) | Random (n=500) | Shuffled (n=500) |
|---|---|---|---|---|
| has_ori | 63.8% | — | — | — |
| has_selection_marker | 65.2% | — | — | — |
| has_promoter | 83.0% | — | — | — |
| has_terminator | 60.1% | — | — | — |
| has_cds | 85.0% | — | — | — |
| **plausibility_pass** | **59.6%** | **95.0%** | **0.0%** | **0.0%** |
| mean n_features | 19.2 | — | — | — |
| mean coverage_frac | 50.1% | — | — | — |

### Tier 5: Architecture
- CDS with promoter context (within 500bp): 43.0%
- CDS with terminator context: 8.6%
- Mean origins per plasmid: 1.6
- Mean selection markers: 1.8
- Mean overlapping features: 5.3

### Memorization
- **Zero hits** at containment threshold 0.3 — no evidence of training set memorization

### Discriminator
- LightGBM AUC: **0.937** (real vs generated)
- Top features by importance: length (5627), ORF density (3533), 6-mer diversity (1735), GC (1419)
- SHAP summary plot saved

**16:45** — Baseline metrics computed: random plausibility 0%, shuffled 0%, real Addgene 95%. Generated model at 59.6% is well above floor and approaching ceiling.

**17:30** — Diversity metrics computed. Model produces comparable diversity to real Addgene plasmids — no mode collapse.

| Metric | Generated | Addgene-500 |
|---|---|---|
| 6-mer JSD (mean pairwise) | 0.352 | 0.334 |
| 6-mer cosine distance | 0.288 | 0.267 |
| 6-mer Jaccard similarity | 0.778 | 0.793 |
| Near-identical pairs (JSD<0.01) | 0.0% | 0.0% |
| Pairs with Jaccard >0.9 | 25.3% | 28.3% |

High Jaccard is expected — real plasmids share extensive backbone sequences (amp resistance, ColE1 ori, etc.). Generated plasmids are slightly more diverse than real ones on all k-mer metrics.

**17:45** — Extra metrics computed: prompt fidelity, codon usage, GC skew, ViennaRNA MFE (running).

### Prompt Fidelity (Conditional Accuracy)
Overall: **23.5%** of requested features detected by pLannotate in generated sequences.

| Category | Hit Rate | Found/Requested |
|---|---|---|
| AMR | 39.2% | 538/1373 |
| PROM | 38.5% | 1090/2828 |
| REPORTER | 27.1% | 95/350 |
| ORI | 19.9% | 339/1701 |
| ELEM | 15.9% | 634/3991 |
| TAG | 1.1% | 2/179 |

Note: pLannotate BLAST may undercount — short features (tags, some elements) and protein-coding features detected better by Smith-Waterman (see earlier alignment eval). The 23.5% is a conservative lower bound.

### Codon Usage
- 3,376 predicted ORFs, 579,368 total codons
- JSD vs E. coli codon frequencies: **0.131** (low divergence = realistic codon usage)

### GC Skew
- Generated mean variation: **0.0566** (reference: 0.0593)
- Very close to real Addgene plasmids — model captures characteristic GC skew patterns

### ViennaRNA MFE Density
- Reference (Addgene-500): DNA MFE density = -0.151 kcal/mol/nt, RNA = -0.327
- Generated: computing (windowed approach for sequences >5kb)...
