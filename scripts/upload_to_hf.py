"""Upload pretrained PlasmidLM checkpoint to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py /path/to/checkpoint --repo McClain/PlasmidLM
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

MODEL_CARD = r"""---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- biology
- genomics
- dna
- plasmid
- synthetic-biology
- causal-lm
- protein-engineering
datasets:
- custom
pipeline_tag: text-generation
model-index:
- name: PlasmidLM
  results:
  - task:
      type: text-generation
      name: Plasmid DNA Generation
    metrics:
    - name: Eval Loss
      type: loss
      value: 0.093
    - name: Token Accuracy
      type: accuracy
      value: 0.961
---

# PlasmidLM

A 17M-parameter transformer language model for conditional generation of synthetic plasmid DNA sequences.

## Model Description

PlasmidLM generates plasmid DNA sequences conditioned on functional component specifications. Given a prompt specifying desired elements (antibiotic resistance genes, origins of replication, promoters, reporters, etc.), it autoregressively generates a complete DNA sequence containing those elements.

**Architecture**: LLaMA-style transformer decoder with RoPE, RMSNorm, and GELU activations.

| Parameter | Value |
|-----------|-------|
| Parameters | 17M |
| Hidden size | 384 |
| Layers | 10 |
| Attention heads | 8 |
| Context length | 16,384 tokens |
| Vocabulary | 120 tokens |

The vocabulary consists of 5 DNA bases (A, T, C, G, N), control tokens (BOS, EOS, SEP, PAD, UNK), and ~100 categorical tokens representing functional plasmid components (e.g., `<AMR_KANAMYCIN>`, `<ORI_COLE1>`, `<PROM_T7>`).

## Training

Pretrained with causal language modeling on ~108K plasmid sequences derived from the [Addgene](https://www.addgene.org/) repository, annotated with functional components via [pLannotate](https://github.com/barricklab/pLannotate).

- **Steps**: 15,000
- **Epochs**: ~2.3
- **Eval loss**: 0.093
- **Token accuracy**: 96.1%
- **Optimizer**: AdamW
- **Precision**: bf16

## Intended Use

This is a **base pretrained model**. It has learned the statistical patterns of plasmid DNA sequences and their relationship to categorical component tokens. It can be used for:

- **Direct generation**: Prompt with component tokens to generate plasmid sequences
- **Fine-tuning**: Post-train with reinforcement learning (GRPO/PPO) to improve motif placement accuracy
- **Embeddings**: Use hidden states as learned representations of plasmid sequences
- **Research**: Study the learned structure of synthetic DNA

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("McClain/PlasmidLM", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("McClain/PlasmidLM", trust_remote_code=True)

# Generate a plasmid with kanamycin resistance and ColE1 origin
prompt = "<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=4096, temperature=0.8, do_sample=True)
sequence = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(sequence)
```

## Input Format

```
<BOS><TOKEN1><TOKEN2>...<SEP>
```

The model generates DNA bases (A/T/C/G) after the `<SEP>` token until it produces `<EOS>` or hits the maximum length.

## Component Categories

| Category | Examples | Count |
|----------|----------|-------|
| Antibiotic Resistance (AMR) | Kanamycin, Ampicillin, Chloramphenicol, ... | 11 |
| Origin of Replication (ORI) | ColE1, F1, P15A, pSC101, SV40, ... | 7 |
| Promoter (PROM) | CMV, T7, U6, EF1a, CAG, ... | 11 |
| Reporter | EGFP, mCherry, YFP, NanoLuc, ... | 6 |
| Vector Type (VEC) | Lentiviral, CRISPR, Bacterial, AAV, ... | 10 |
| Other | Tags, elements, species, backbones | ~55 |

## Post-Training (RL) — Sampling Experiments

We are actively running REINFORCE with KL penalty to improve motif placement accuracy. The policy is trained against a Smith-Waterman alignment reward that scores whether generated sequences contain the motifs specified in the prompt.

**Setup**: Ray-based distributed training — GPU policy actor generates completions, CPU workers score with parasail alignment. Curriculum learning ramps `alpha` from 0 (presence-only scoring) to 1 (exact-match scoring) over 2000 steps.

**Key findings from sampling runs**:

| Run | LR | KL coef | Log probs | Result |
|-----|-----|---------|-----------|--------|
| v1 | 1e-4 | 0.1 | sum over tokens | KL explosion at step ~120, collapsed to reward=0 |
| v2 | 5e-5 | 0.5 | per-token mean | Collapsed at step ~150 — short-sequence mode collapse |
| v3 | 1e-5 | 1.0 | per-token mean + EMA baseline + smooth length reward | Stable through step 250+, ongoing |

**Critical fixes discovered**:
1. **Per-token normalization**: Summing log probs over 1000+ tokens made KL/loss magnitudes ~1000x too large. Switching to per-token mean stabilized gradients.
2. **Smooth length reward**: Hard reward=0 cliff for sequences <100bp caused irrecoverable mode collapse. Replaced with smooth ramp over [0, 500bp].
3. **EMA reward baseline**: Batch-mean baseline gave zero advantage when all rewards collapsed. EMA (decay=0.95) retains memory of previous rewards, providing negative advantage signal to push away from degenerate outputs.

**MLflow tracking**: Experiments logged to Databricks MLflow at experiment path `/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM`.

## Limitations

- This is a **pretrained base model** — it learns sequence statistics but has not been optimized for motif placement accuracy. Post-training with RL significantly improves functional element fidelity.
- Generated sequences are **not experimentally validated**. Always verify computationally (e.g., with pLannotate) and experimentally before synthesis.
- The model was trained on Addgene plasmids, which are biased toward commonly deposited vectors (mammalian expression, bacterial cloning, CRISPR).
- Maximum context of 16K tokens (~16 kbp), which covers most but not all plasmids.

## Citation

```bibtex
@misc{thiel2026plasmidlm,
  title={PlasmidLM: Language Models for Conditional Plasmid DNA Generation},
  author={Thiel, McClain},
  year={2026},
  url={https://huggingface.co/McClain/PlasmidLM}
}
```
"""

# Files to upload from the checkpoint (exclude training artifacts)
UPLOAD_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "special_tokens.txt",
    "configuration_plasmid_lm.py",
    "modeling_plasmid_lm.py",
    "tokenization_plasmid_lm.py",
]


def main():
    parser = argparse.ArgumentParser(description="Upload PlasmidLM to HuggingFace Hub")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint directory")
    parser.add_argument("--repo", default="McClain/PlasmidLM", help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Verify all required files exist
    missing = [f for f in UPLOAD_FILES if not (checkpoint / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in checkpoint: {missing}")

    print(f"Checkpoint: {checkpoint}")
    print(f"Repo: {args.repo}")
    print(f"Files to upload:")
    total_size = 0
    for f in UPLOAD_FILES:
        size = (checkpoint / f).stat().st_size
        total_size += size
        print(f"  {f:40s} {size / 1024 / 1024:8.2f} MB")
    print(f"  {'README.md':40s} {len(MODEL_CARD) / 1024:8.2f} KB")
    print(f"  Total: {total_size / 1024 / 1024:.1f} MB")

    if args.dry_run:
        print("\nDry run -- exiting.")
        return

    # Create repo
    api = HfApi()
    url = create_repo(args.repo, exist_ok=True, private=args.private)
    print(f"\nRepo created/found: {url}")

    # Stage files in a temp directory to add the README
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy model files
        for f in UPLOAD_FILES:
            shutil.copy2(checkpoint / f, tmpdir / f)

        # Write model card
        (tmpdir / "README.md").write_text(MODEL_CARD)

        # Upload
        print("Uploading...")
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=args.repo,
            commit_message="Upload PlasmidLM pretrained checkpoint (v4, step 15000)",
        )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
