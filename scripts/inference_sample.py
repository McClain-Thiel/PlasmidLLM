"""Sample inference from a trained model on validation prompts using vLLM."""

from __future__ import annotations

import argparse
import csv

import pyarrow.parquet as pq
import torch
from vllm import LLM, SamplingParams

from plasmid_llm.tokenizer import PlasmidTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples for validation prompts using vLLM"
    )
    parser.add_argument("--hf-model", required=True, help="Path to HF model directory")
    parser.add_argument("--parquet", required=True, help="Path to training_pairs.parquet")
    parser.add_argument("--vocab", required=True, help="Path to vocab JSON")
    parser.add_argument("--output", default="inference_samples.csv")
    parser.add_argument("--n", type=int, default=20, help="Number of val prompts to sample")
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling (-1 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split fraction")
    parser.add_argument("--data-seed", type=int, default=42, help="Seed used for train/val split")
    args = parser.parse_args()

    tokenizer = PlasmidTokenizer(args.vocab)

    # Load data and reproduce val split
    print(f"Loading data: {args.parquet}")
    table = pq.read_table(args.parquet)
    col_names = table.column_names
    prompts = table.column("token_prompt").to_pylist()
    completion_col = "token_completion" if "token_completion" in col_names else "sequence"
    completions = table.column(completion_col).to_pylist()

    n_total = len(prompts)
    n_val = int(n_total * args.val_split)
    torch.manual_seed(args.data_seed)
    indices = torch.randperm(n_total).tolist()
    val_indices = indices[-n_val:]

    # Sample n prompts from val set
    torch.manual_seed(args.seed)
    sample_indices = torch.randperm(len(val_indices))[: args.n].tolist()
    selected = [val_indices[i] for i in sample_indices]

    # Prepare prompt texts (with SEP appended)
    prompt_texts = [prompts[idx] + "<SEP>" for idx in selected]

    # Tokenize all prompts
    prompt_token_ids = [tokenizer.encode(p) for p in prompt_texts]

    # Initialize vLLM engine
    print(f"Loading vLLM model: {args.hf_model}")
    llm = LLM(model=args.hf_model, trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else -1,
        seed=args.seed,
    )

    # Batch generate all prompts at once
    print(f"Generating {len(prompt_token_ids)} samples...")
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    # Collect results
    results = []
    for i, (idx, output) in enumerate(zip(selected, outputs)):
        generated_ids = output.outputs[0].token_ids
        generated_text = tokenizer.decode(list(generated_ids))

        results.append({
            "idx": idx,
            "prompt": prompts[idx],
            "true_completion": completions[idx],
            "generated": generated_text,
            "true_len": len(completions[idx]),
            "gen_len": len(generated_text),
        })
        print(f"[{i + 1}/{args.n}] prompt={prompts[idx][:60]}... gen_len={len(generated_text)}")

    # Write CSV
    print(f"\nWriting {len(results)} samples to {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Done!")


if __name__ == "__main__":
    main()
