# Eval Suite Log

## 2026-04-08

**11:30** — Phase 0 started. Installed sourmash 4.3, minimap2 2.30, lightgbm 4.6, shap 0.49, sklearn 1.7 into plannotate conda env. Fixed sourmash pkg_resources issue (setuptools<70).

**11:45** — Downloaded Addgene-500 reference panel from HF bucket `McClain/PlasmidRL/reference/`. 500 plasmids, 418 not in training set. Pre-computed: length, GC, longest ORF, MFE density, 3-mer freqs.

**11:50** — Pinned tool versions in `eval/config/environment.yml`. pLannotate 1.2.2 (snapgene DB 2021-11), prodigal 2.6.3, BLAST+ 2.17.0.

**11:55** — Built `eval/config/feature_categories.yaml` mapping pLannotate feature types to Tier 3 booleans (origin, selection_marker, promoter, terminator, cds).

**12:00** — Built training-set sourmash signature: 20,644 unique plasmids, k=31 scaled=100. Cached at `eval/reference/training_sigs.zip`.

**12:05** — Building negative controls (random GC-matched, dinucleotide-shuffled, real held-out). Writing generation + annotation pipeline scripts.
