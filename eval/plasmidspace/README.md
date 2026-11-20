---
title: PlasmidSpace
emoji: "\U0001F9EC"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
python_version: "3.10"
models:
  - McClain/PlasmidLM-kmer6-GRPO-plannotate
---

# PlasmidSpace

Design synthetic plasmids from natural language using
[PlasmidLM](https://huggingface.co/McClain/PlasmidLM-kmer6-GRPO-plannotate).

## How it works

1. **Pick a preset** or **describe** your plasmid in plain English
2. **Map to Tokens** — Claude translates your description into structured component tokens
3. **Generate** — PlasmidLM (19.3M param, GRPO-trained) generates a DNA sequence conditioned on those tokens
4. **Annotate** — pLannotate verifies which functional elements are present and renders a plasmid map
