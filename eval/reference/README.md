# Evaluation reference artifacts

The compact derived summaries remain in Git:

- `addgene_500_3mer_freqs.json`
- `addgene_reference_metrics.csv`

The source panel and Sourmash index are external binary artifacts:

| file | bytes | SHA-256 |
|---|---:|---|
| `addgene_500.fasta` | 3,749,506 | `1b3189576d9d0487a9ece911628580fc7b618d600ba758fb2299c5a77a5018b5` |
| `addgene_reference_500.csv` | 3,751,628 | `66d1402d1b09a78f31b75f0ec435c320aabbb61a73e11203a82f3c5f59fa668b` |
| `training_sigs.zip` | 1,134,802 | `7a50e5dd3339ceb6ce795648bb9101cb9bc1c89f4c31afde0aafaa725ca664d8` |

They are stored under:

```text
hf://buckets/<org>/PlasmidLMEval/neurips_2026_rebuttal/
  s3/neurips_rebuttal/repo_release/reference/
```

Download them into this directory before running the standing evaluation
pipeline. Verify their hashes against this table or the archive-wide
`lineage/SHA256SUMS` file.
