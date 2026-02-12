"""Generate plasmid sequences from a trained model checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from plasmid_llm.models import build_model
from plasmid_llm.tokenizer import PlasmidTokenizer


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load model and tokenizer from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    tokenizer = PlasmidTokenizer(cfg.data.vocab_path)
    model = build_model(cfg.model, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, cfg


def main():
    parser = argparse.ArgumentParser(description="Generate plasmid sequences")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", default="<BOS><SEP>", help="Tag prompt (e.g. '<BOS><AMR_KANAMYCIN><ORI_F1><SEP>')")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer, cfg = load_model(args.checkpoint, args.device)
    print(f"Loaded {cfg.model.arch} model ({sum(p.numel() for p in model.parameters()):,} params)")

    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids] * args.num_samples, device=args.device)

    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.num_samples} sequence(s)...\n")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    for i in range(args.num_samples):
        seq = tokenizer.decode(output_ids[i].tolist())
        print(f"--- Sample {i + 1} ---")
        print(seq)
        print()


if __name__ == "__main__":
    main()
