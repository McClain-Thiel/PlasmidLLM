# scripts

Standalone utility scripts. These are not part of the training pipeline — see `pretraining/` and `post_training/` for training entry points.

## Files

### `generate.py`

Generate plasmid sequences from a trained checkpoint using vLLM.

```bash
python scripts/generate.py \
    --hf-model output/pretraining/final \
    --vocab output/pretraining/final/vocab.json \
    --prompt "<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>" \
    --num-samples 5 \
    --temperature 1.0 \
    --max-new-tokens 8000
```

Takes a token prompt describing the desired plasmid components and generates one or more DNA sequences. Uses vLLM for fast batched inference.

### `build_motif_registry.py`

Build the motif registry that maps categorical tokens to canonical reference sequences.

```bash
python scripts/build_motif_registry.py
```

Queries plannotate's BLAST/Diamond/Infernal databases to find representative sequences for each categorical token (`<AMR_*>`, `<ORI_*>`, `<PROM_*>`, etc.). Outputs both `motif_registry.json` and `motif_registry.parquet`. The motif registry is required by the post-training reward function to verify that generated sequences contain the correct motifs.

### `upload_to_hf.py`

Upload a trained checkpoint to HuggingFace Hub.

```bash
python scripts/upload_to_hf.py /path/to/checkpoint --repo McClain/PlasmidLM
```

Packages the model weights, config, tokenizer, and a generated model card into a HuggingFace repository. Handles vocab file copying and README generation automatically.
