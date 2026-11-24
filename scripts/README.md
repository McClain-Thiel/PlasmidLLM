# scripts

Standalone CLI utilities (separate from the training loops in `pretraining/` and `post_training/`).

## Files

### `generate.py`

Generate plasmid sequences from a local or Hugging Face checkpoint using `vLLM`:

```bash
python scripts/generate.py \
    --hf-model McClain/PlasmidLM-kmer6-GRPO-plannotate \
    --prompt "<BOS><VEC_BACTERIAL><AMR_KANAMYCIN><ORI_COLE1><PROM_T7><SEQ>" \
    --num-samples 4 \
    --temperature 0.3
```

### `benchmark_inference.py`

Measure autoregressive generation throughput and token/bp latency for released checkpoints.

### `build_motif_registry.py`

Build the `660`-entry curated BLAST/ORF motif registry (`motif_registry_combined.parquet`) bridging categorical special tokens to `pLannotate` references (SnapGene, FPbase, SwissProt, Rfam):

```bash
PLASMID_ANNOTATION_ROOT=/path/to/addgene_clean python scripts/build_motif_registry.py
```

### `upload_to_hf.py`

Package checkpoint weights, modeling/tokenizer files, config, and model card into a Hugging Face Hub repository:

```bash
python scripts/upload_to_hf.py /path/to/checkpoint --repo McClain/PlasmidLM-kmer6
```
