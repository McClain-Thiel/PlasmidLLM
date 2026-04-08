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
