"""Upload GRPO pLannotate-trained PlasmidLM checkpoint to HuggingFace Hub."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

CHECKPOINT = Path("/opt/dlami/nvme/eval_checkpoints/grpo_plannotate/step_800")
REPO_ID = "McClain/PlasmidLM-kmer6-GRPO-plannotate"

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
- reinforcement-learning
- grpo
- custom_code
base_model: McClain/PlasmidLM-kmer6
datasets:
- custom
pipeline_tag: text-generation
model-index:
- name: PlasmidLM-kmer6-GRPO-plannotate
  results:
  - task:
      type: text-generation
      name: Plasmid DNA Generation
    metrics:
    - name: pLannotate Hit Rate (t=0.3)
      type: accuracy
      value: 0.680
    - name: pLannotate Hit Rate (t=0.5)
      type: accuracy
      value: 0.688
---

# PlasmidLM-kmer6-GRPO-plannotate

A 19.3M parameter plasmid DNA generation model, post-trained with **GRPO (Group Relative Policy Optimization)** using [pLannotate](https://github.com/barricklab/pLannotate) biological annotations as a reward signal. Fine-tuned from [McClain/PlasmidLM-kmer6](https://huggingface.co/McClain/PlasmidLM-kmer6).

## What's New vs Base Model

This model was post-trained with reinforcement learning to improve the biological accuracy of generated plasmid sequences. Instead of only learning sequence statistics, the model was optimized to produce sequences where requested functional elements (antibiotic resistance genes, origins of replication, promoters, etc.) are **verifiably present** when analyzed by the pLannotate annotation tool.

| Metric | Base Model | GRPO-plannotate | Improvement |
|--------|-----------|-----------------|-------------|
| **Overall Hit Rate** | 59.2% | **68.0%** | +8.8pp |
| AMR (Antibiotic Resistance) | 63.8% | **70.7%** | +6.9pp |
| ORI (Origin of Replication) | 73.6% | **80.6%** | +7.0pp |
| PROM (Promoters) | 66.9% | **72.4%** | +5.5pp |
| ELEM (Other Elements) | 52.9% | 51.0% | -1.9pp |
| REPORTER | 17.6% | 17.6% | 0pp |

*Evaluated on 50 held-out validation prompts with best-of-3 sampling at temperature 0.3.*

## Model Details

| Property | Value |
|---|---|
| Parameters | 19.3M |
| Architecture | Transformer decoder (dense MLP), LLaMA-style |
| Hidden size | 384 |
| Layers | 10 |
| Attention heads | 8 |
| Intermediate size | 1,536 |
| Max sequence length | 16,384 tokens |
| Tokenizer | k-mer (k=6, stride=3) |
| Vocab size | 4,208 |

## Training

### Pretraining
- **Data**: ~100K plasmid sequences from Addgene, tokenized with k-mer (k=6, stride=3)
- **Base checkpoint**: [McClain/PlasmidLM-kmer6](https://huggingface.co/McClain/PlasmidLM-kmer6) (65K steps, eval loss 0.129)

### GRPO Post-Training
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward**: pLannotate biological annotation — generated sequences are annotated with pLannotate, and the reward reflects how many requested functional elements are found with >= 95% sequence identity
- **Steps**: 800
- **Infrastructure**: Anyscale (Ray-based distributed training)
- **W&B Run**: [mcclain/PlasmidLLM/runs/sil7t16f](https://wandb.ai/mcclain/PlasmidLLM/runs/sil7t16f)

### Temperature Sensitivity

| Temperature | Hit Rate |
|---|---|
| 0.1 | 66.8% |
| 0.3 | 68.0% |
| 0.5 | 68.8% |
| 0.7 | 50.7%* |
| 1.0 | 30.1%* |

*Evaluated on step_100 checkpoint; step_800 performs better at all temperatures. Recommended range: **0.3 - 0.5**.*

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "McClain/PlasmidLM-kmer6-GRPO-plannotate"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Generate a plasmid with kanamycin resistance, ColE1 origin, and T7 promoter
prompt = "<BOS> <AMR_KANAMYCIN> <ORI_COLE1> <PROM_T7> <SEP>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3000,
        temperature=0.3,
        do_sample=True,
        top_k=50,
    )

sequence = tokenizer.decode(outputs[0].tolist())
print(sequence)

# Extract just the DNA sequence
import re
dna = re.sub(r"<[^>]+>", "", sequence.upper())
dna = re.sub(r"[^ATGCN]", "", dna)
print(f"Generated {len(dna)} bp plasmid sequence")
```

## Input Format

```
<BOS> <TOKEN1> <TOKEN2> ... <SEP>
```

The model generates k-mer encoded DNA after `<SEP>` until `<EOS>` or max length. Spaces between tokens are optional but recommended.

## Available Component Tokens

| Category | Tokens |
|----------|--------|
| **Antibiotic Resistance (AMR)** | AMPICILLIN, KANAMYCIN, CHLORAMPHENICOL, SPECTINOMYCIN, GENTAMICIN, PUROMYCIN, HYGROMYCIN, BLASTICIDIN, NEOMYCIN, ZEOCIN, TETRACYCLINE |
| **Origin of Replication (ORI)** | COLE1, F1, P15A, PSC101, SV40, 2MU, RSF |
| **Promoter (PROM)** | CMV, T7, U6, EF1A, CAG, LAC, SV40, AMPR, RSV, SP6, T3 |
| **Reporter** | EGFP, GFP, MCHERRY, YFP, NANOLUC, LUCIFERASE |
| **Tags** | HIS, FLAG, MYC, HA, GST, NLS |
| **Elements (ELEM)** | WPRE, POLYA_BGH, POLYA_SV40, CMV_ENHANCER, MCS, LTR_5, LTR_3, PSI, CPPT, AAV_ITR, GRNA_SCAFFOLD |

Format: `<CATEGORY_NAME>`, e.g. `<AMR_KANAMYCIN>`, `<ORI_COLE1>`, `<PROM_T7>`

## Limitations

- Generated sequences are **not experimentally validated**. Always verify computationally (e.g., with pLannotate) and experimentally before synthesis.
- The model was trained on Addgene plasmids, biased toward commonly deposited vectors.
- Reporter and Tag categories have low hit rates and may need further RL training.
- Maximum context of 16K tokens.

## Citation

```bibtex
@misc{thiel2026plasmidlm,
  title={PlasmidLM: Language Models for Conditional Plasmid DNA Generation with Reinforcement Learning},
  author={Thiel, McClain},
  year={2026},
  url={https://huggingface.co/McClain/PlasmidLM-kmer6-GRPO-plannotate}
}
```
"""

UPLOAD_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "tokenizer_config.json",
    "tokenization_kmer.py",
    "configuration_plasmid_lm.py",
    "modeling_plasmid_lm.py",
    "moe.py",
]


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    missing = [f for f in UPLOAD_FILES if not (CHECKPOINT / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Repo: {REPO_ID}")
    print("Files:")
    total = 0
    for f in UPLOAD_FILES:
        size = (CHECKPOINT / f).stat().st_size
        total += size
        print(f"  {f:45s} {size / 1024 / 1024:8.2f} MB")
    print(f"  Total: {total / 1024 / 1024:.1f} MB")

    api = HfApi()
    url = create_repo(REPO_ID, exist_ok=True, private=False)
    print(f"\nRepo: {url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for f in UPLOAD_FILES:
            shutil.copy2(CHECKPOINT / f, tmpdir / f)
        (tmpdir / "README.md").write_text(MODEL_CARD)

        print("Uploading...")
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=REPO_ID,
            commit_message="Upload PlasmidLM-kmer6-GRPO-plannotate (step 800, pLannotate reward)",
        )

    print(f"\nDone! https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
