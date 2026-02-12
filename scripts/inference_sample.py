"""Sample inference from a trained model on validation prompts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.tokenizer import PlasmidTokenizer
from plasmid_llm.models import build_model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", default="inference_samples.csv")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    # Load tokenizer
    print(f"Loading vocab: {args.vocab}")
    tokenizer = PlasmidTokenizer(args.vocab)

    # Build model and load weights
    print(f"Building {cfg.model.arch} model...")
    model = build_model(cfg.model, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load data and get val split
    print(f"Loading data: {args.parquet}")
    table = pq.read_table(args.parquet)
    col_names = table.column_names
    prompts = table.column("token_prompt").to_pylist()
    completion_col = "token_completion" if "token_completion" in col_names else "sequence"
    completions = table.column(completion_col).to_pylist()

    # Reproduce val split
    n_total = len(prompts)
    n_val = int(n_total * cfg.data.val_split)
    torch.manual_seed(cfg.data.seed)
    indices = torch.randperm(n_total).tolist()
    val_indices = indices[-n_val:]

    # Sample n prompts from val set
    torch.manual_seed(args.seed)
    sample_indices = torch.randperm(len(val_indices))[: args.n].tolist()
    selected = [val_indices[i] for i in sample_indices]

    sep_id = tokenizer.sep_token_id
    sep_token = "<SEP>"

    results = []
    for i, idx in enumerate(selected):
        prompt_text = prompts[idx]
        true_completion = completions[idx]

        # Encode prompt + SEP
        prompt_with_sep = prompt_text + sep_token
        input_ids = tokenizer.encode(prompt_with_sep)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        # Decode generated part only
        generated_ids = output_ids[0, len(input_ids) :].tolist()
        generated_text = tokenizer.decode(generated_ids)

        # Truncate true completion for display
        true_short = true_completion[:200] + "..." if len(true_completion) > 200 else true_completion
        gen_short = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text

        results.append({
            "idx": idx,
            "prompt": prompt_text,
            "true_completion": true_short,
            "generated": gen_short,
            "true_len": len(true_completion),
            "gen_len": len(generated_text),
        })

        print(f"[{i+1}/{args.n}] prompt={prompt_text[:60]}... gen_len={len(generated_text)}")

    # Write CSV
    print(f"\nWriting {len(results)} samples to {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Done!")


if __name__ == "__main__":
    main()
