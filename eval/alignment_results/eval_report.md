# PlasmidLLM Alignment Evaluation Report

Generated: 2026-03-06 00:48:02
Cutoff: 70% score ratio

## Summary

| Model | Temp | Hit Rate | Precision | TP | FN | FP | EOS% | Avg Len | TPS |
|-------|------|----------|-----------|----|----|----|----- |---------|-----|
| PlasmidLM-kmer6 | 0.3 | 62.4% | 94.0% | 245 | 139 | 0 | 56% | 7497 | 897 |
| PlasmidLM-kmer6 | 0.5 | 53.5% | 94.0% | 202 | 182 | 0 | 72% | 7021 | 876 |
| PlasmidLM-kmer6 | 0.7 | 52.4% | 92.0% | 200 | 184 | 0 | 66% | 7358 | 919 |
| PlasmidLM-kmer6 | 0.9 | 45.8% | 82.0% | 177 | 207 | 0 | 68% | 6699 | 920 |
| PlasmidLM-kmer6 | 1.0 | 47.3% | 84.0% | 173 | 211 | 0 | 60% | 7103 | 861 |
| PlasmidLM-kmer6-MoE | 0.3 | 48.2% | 76.0% | 187 | 197 | 0 | 54% | 7496 | 820 |
| PlasmidLM-kmer6-MoE | 0.5 | 46.4% | 86.0% | 187 | 197 | 0 | 58% | 7234 | 849 |
| PlasmidLM-kmer6-MoE | 0.7 | 50.6% | 84.0% | 201 | 183 | 0 | 68% | 6900 | 842 |
| PlasmidLM-kmer6-MoE | 0.9 | 49.8% | 78.0% | 203 | 181 | 0 | 70% | 6595 | 782 |
| PlasmidLM-kmer6-MoE | 1.0 | 42.4% | 80.0% | 172 | 212 | 0 | 68% | 6460 | 761 |

## Per-Category Breakdown (best temp per model)

### PlasmidLM-kmer6 (temp=0.3)

| Category | Requested | TP | FN | FP | Hit Rate |
|----------|-----------|----|----|----|----- ---|
| AMR | 58 | 41 | 17 | 0 | 70.7% |
| ELEM | 102 | 54 | 48 | 0 | 52.9% |
| ORI | 72 | 58 | 14 | 0 | 80.6% |
| PROM | 127 | 85 | 42 | 0 | 66.9% |
| REPORTER | 17 | 5 | 12 | 0 | 29.4% |
| TAG | 8 | 2 | 6 | 0 | 25.0% |
| **TOTAL** | **384** | **245** | **139** | **0** | **63.8%** |

### PlasmidLM-kmer6-MoE (temp=0.7)

| Category | Requested | TP | FN | FP | Hit Rate |
|----------|-----------|----|----|----|----- ---|
| AMR | 58 | 33 | 25 | 0 | 56.9% |
| ELEM | 102 | 42 | 60 | 0 | 41.2% |
| ORI | 72 | 45 | 27 | 0 | 62.5% |
| PROM | 127 | 71 | 56 | 0 | 55.9% |
| REPORTER | 17 | 9 | 8 | 0 | 52.9% |
| TAG | 8 | 1 | 7 | 0 | 12.5% |
| **TOTAL** | **384** | **201** | **183** | **0** | **52.3%** |

## Per-Prompt Details (first 15, best temp)

### PlasmidLM-kmer6 (temp=0.3)

- **[80%]** TP=4 FN=1 FP=0 len=4779 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <PROM_LAC>
  - FN: <PROM_T7>(0.42)
- **[83%]** TP=5 FN=1 FP=0 len=6885 eos=True
  - TP: <AMR_AMPICILLIN> <ELEM_POLYA_SV40> <ORI_COLE1> <PROM_AMPR> <TAG_NLS>
  - FN: <REPORTER_EGFP>(0.04)
- **[PERFECT]** TP=4 FN=0 FP=0 len=9003 eos=False
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <REPORTER_GFP>
- **[70%]** TP=7 FN=3 FP=0 len=9003 eos=False
  - TP: <AMR_AMPICILLIN> <ELEM_LTR_5> <ELEM_PSI> <ELEM_WPRE> <ORI_COLE1> <PROM_AMPR> <PROM_LAC>
  - FN: <ELEM_CMV_ENHANCER>(0.04), <ELEM_CPPT>(0.19), <PROM_CMV>(0.23)
- **[PERFECT]** TP=4 FN=0 FP=0 len=6381 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <PROM_LAC>
- **[PERFECT]** TP=3 FN=0 FP=0 len=3858 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR>
- **[50%]** TP=8 FN=8 FP=0 len=7422 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <ORI_F1> <PROM_AMPR> <PROM_LAC> <PROM_T3> <PROM_T7> <REPORTER_EGFP>
  - FN: <ELEM_CPPT>(0.16), <ELEM_LTR_3>(0.05), <ELEM_LTR_5>(0.07), <ELEM_POLYA_SV40>(0.13), <ELEM_PSI>(0.10), <ELEM_WPRE>(0.03), <ORI_SV40>(0.09), <PROM_RSV>(0.05)
- **[80%]** TP=4 FN=1 FP=0 len=7098 eos=True
  - TP: <AMR_AMPICILLIN> <ELEM_PSI> <ORI_COLE1> <PROM_AMPR>
  - FN: <AMR_BLASTICIDIN>(0.07)
- **[80%]** TP=4 FN=1 FP=0 len=5985 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <PROM_LAC>
  - FN: <TAG_GST>(0.05)
- **[PERFECT]** TP=4 FN=0 FP=0 len=9003 eos=False
  - TP: <AMR_KANAMYCIN> <ELEM_MCS> <ORI_COLE1> <PROM_LAC>
- **[75%]** TP=15 FN=5 FP=0 len=9003 eos=False
  - TP: <AMR_AMPICILLIN> <AMR_ZEOCIN> <ELEM_CMV_ENHANCER> <ELEM_LTR_3> <ELEM_LTR_5> <ELEM_POLYA_BGH> <ELEM_POLYA_SV40> <ELEM_PSI> <ORI_COLE1> <ORI_F1> <ORI_SV40> <PROM_AMPR> <PROM_CMV> <PROM_LAC> <PROM_SV40>
  - FN: <ELEM_CPPT>(0.14), <ELEM_GRNA_SCAFFOLD>(0.18), <ELEM_WPRE>(0.04), <PROM_EF1A>(0.06), <PROM_U6>(0.16)
- **[67%]** TP=2 FN=1 FP=0 len=4878 eos=True
  - TP: <AMR_AMPICILLIN> <PROM_AMPR>
  - FN: <ORI_RSF>(0.67)
- **[50%]** TP=4 FN=4 FP=0 len=8991 eos=False
  - TP: <ELEM_POLYA_SV40> <ORI_COLE1> <ORI_F1> <PROM_AMPR>
  - FN: <AMR_AMPICILLIN>(0.04), <ELEM_AAV_ITR>(0.11), <ELEM_WPRE>(0.03), <REPORTER_EGFP>(0.04)
- **[33%]** TP=2 FN=4 FP=0 len=9003 eos=False
  - TP: <ELEM_POLYA_BGH> <ORI_COLE1>
  - FN: <AMR_AMPICILLIN>(0.25), <AMR_PUROMYCIN>(0.05), <PROM_AMPR>(0.17), <PROM_EF1A>(0.06)
- **[29%]** TP=2 FN=5 FP=0 len=8940 eos=False
  - TP: <ORI_COLE1> <PROM_AMPR>
  - FN: <AMR_AMPICILLIN>(0.24), <AMR_BLASTICIDIN>(0.06), <ELEM_PSI>(0.10), <PROM_LAC>(0.32), <PROM_T3>(0.42)

### PlasmidLM-kmer6-MoE (temp=0.7)

- **[0%]** TP=0 FN=5 FP=0 len=7245 eos=True
  - FN: <AMR_AMPICILLIN>(0.04), <ORI_COLE1>(0.12), <PROM_AMPR>(0.19), <PROM_LAC>(0.34), <PROM_T7>(0.42)
- **[67%]** TP=4 FN=2 FP=0 len=6429 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <REPORTER_EGFP>
  - FN: <ELEM_POLYA_SV40>(0.15), <TAG_NLS>(0.58)
- **[75%]** TP=3 FN=1 FP=0 len=8256 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR>
  - FN: <REPORTER_GFP>(0.35)
- **[90%]** TP=9 FN=1 FP=0 len=9003 eos=False
  - TP: <AMR_AMPICILLIN> <ELEM_CPPT> <ELEM_LTR_5> <ELEM_PSI> <ELEM_WPRE> <ORI_COLE1> <PROM_AMPR> <PROM_CMV> <PROM_LAC>
  - FN: <ELEM_CMV_ENHANCER>(0.04)
- **[0%]** TP=0 FN=4 FP=0 len=8298 eos=True
  - FN: <AMR_AMPICILLIN>(0.04), <ORI_COLE1>(0.16), <PROM_AMPR>(0.20), <PROM_LAC>(0.47)
- **[PERFECT]** TP=3 FN=0 FP=0 len=5007 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR>
- **[56%]** TP=9 FN=7 FP=0 len=6477 eos=True
  - TP: <AMR_AMPICILLIN> <ELEM_POLYA_SV40> <ORI_COLE1> <ORI_F1> <ORI_SV40> <PROM_AMPR> <PROM_LAC> <PROM_T7> <REPORTER_EGFP>
  - FN: <ELEM_CPPT>(0.14), <ELEM_LTR_3>(0.05), <ELEM_LTR_5>(0.08), <ELEM_PSI>(0.10), <ELEM_WPRE>(0.03), <PROM_RSV>(0.05), <PROM_T3>(0.47)
- **[60%]** TP=3 FN=2 FP=0 len=7077 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR>
  - FN: <AMR_BLASTICIDIN>(0.06), <ELEM_PSI>(0.11)
- **[80%]** TP=4 FN=1 FP=0 len=4299 eos=True
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <PROM_LAC>
  - FN: <TAG_GST>(0.04)
- **[0%]** TP=0 FN=4 FP=0 len=7719 eos=True
  - FN: <AMR_KANAMYCIN>(0.05), <ELEM_MCS>(0.21), <ORI_COLE1>(0.13), <PROM_LAC>(0.35)
- **[70%]** TP=14 FN=6 FP=0 len=8175 eos=True
  - TP: <AMR_AMPICILLIN> <ELEM_CPPT> <ELEM_GRNA_SCAFFOLD> <ELEM_LTR_3> <ELEM_LTR_5> <ELEM_POLYA_SV40> <ELEM_PSI> <ORI_COLE1> <ORI_F1> <ORI_SV40> <PROM_AMPR> <PROM_LAC> <PROM_SV40> <PROM_U6>
  - FN: <AMR_ZEOCIN>(0.07), <ELEM_CMV_ENHANCER>(0.05), <ELEM_POLYA_BGH>(0.13), <ELEM_WPRE>(0.04), <PROM_CMV>(0.26), <PROM_EF1A>(0.06)
- **[67%]** TP=2 FN=1 FP=0 len=6180 eos=True
  - TP: <AMR_AMPICILLIN> <PROM_AMPR>
  - FN: <ORI_RSF>(0.67)
- **[75%]** TP=6 FN=2 FP=0 len=3795 eos=True
  - TP: <AMR_AMPICILLIN> <ELEM_POLYA_SV40> <ORI_COLE1> <ORI_F1> <PROM_AMPR> <REPORTER_EGFP>
  - FN: <ELEM_AAV_ITR>(0.11), <ELEM_WPRE>(0.04)
- **[67%]** TP=4 FN=2 FP=0 len=5235 eos=True
  - TP: <AMR_AMPICILLIN> <AMR_PUROMYCIN> <ORI_COLE1> <PROM_AMPR>
  - FN: <ELEM_POLYA_BGH>(0.12), <PROM_EF1A>(0.06)
- **[71%]** TP=5 FN=2 FP=0 len=9003 eos=False
  - TP: <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <PROM_LAC> <PROM_T3>
  - FN: <AMR_BLASTICIDIN>(0.06), <ELEM_PSI>(0.12)
