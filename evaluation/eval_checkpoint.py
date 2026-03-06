"""Evaluate a checkpoint on validation prompts: generate sequences and score component placement.

Usage:
    python scripts/eval_checkpoint.py /path/to/checkpoint --n 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plasmid_llm.models.hf_plasmid_lm import (
    PlasmidKmerTokenizer,
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMTokenizer,
)
from post_training.scorers.alignment import (
    QC_THRESHOLD,
    AlignmentScorer,
    load_motif_lookup,
    parse_hard_tokens,
    score_motif,
)


def generate_sequences(model, tokenizer, prompts, max_new_tokens=4096, temperature=0.8, device="cuda"):
    """Generate one sequence per prompt."""
    model.eval()
    results = []

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode full output (including prompt)
        full_text = tokenizer.decode(output_ids[0].tolist())
        # Extract completion (after SEP)
        completion = full_text.split("<SEP>")[-1] if "<SEP>" in full_text else full_text
        results.append(completion)

    return results


def extract_dna(text):
    """Strip tokens and non-DNA chars."""
    seq = re.sub(r"<[^>]+>", "", text.upper())
    seq = re.sub(r"[^ATGCN]", "", seq)
    return seq


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on validation prompts")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument("--parquet", type=str,
                        default="/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet")
    parser.add_argument("--motif-lookup", type=str,
                        default="/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet")
    parser.add_argument("--n", type=int, default=50, help="Number of val prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.05)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Load model — auto-detect tokenizer type from checkpoint config
    print(f"Loading model: {args.checkpoint}")
    model = PlasmidLMForCausalLM.from_pretrained(str(args.checkpoint)).to(device)
    model_config = PlasmidLMConfig.from_pretrained(str(args.checkpoint))
    if getattr(model_config, "tokenizer_type", "char") == "kmer":
        k = getattr(model_config, "kmer_k", 6)
        stride = getattr(model_config, "kmer_stride", 3)
        tokenizer = PlasmidKmerTokenizer.from_pretrained(str(args.checkpoint), k=k, stride=stride)
        print(f"Using k-mer tokenizer (k={k}, stride={stride})")
    else:
        tokenizer = PlasmidLMTokenizer.from_pretrained(str(args.checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"

    # Load motif lookup
    print(f"Loading motif lookup: {args.motif_lookup}")
    lookup_df = load_motif_lookup(args.motif_lookup)
    scorer = AlignmentScorer(lookup_df=lookup_df)

    # Load val prompts
    print(f"Loading prompts: {args.parquet}")
    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet)
    col = "prompt" if "prompt" in table.column_names else "token_prompt"
    all_prompts = table.column(col).to_pylist()

    # Reproduce val split
    n_total = len(all_prompts)
    n_val = int(n_total * args.val_split)
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    val_indices = indices[:n_val]

    # Sample n prompts
    torch.manual_seed(args.seed)
    sample_idx = torch.randperm(len(val_indices))[:args.n].tolist()
    selected = [val_indices[i] for i in sample_idx]

    prompts = [all_prompts[i] + "<SEP>" for i in selected]

    print(f"\nGenerating {len(prompts)} sequences...")

    # Generate
    completions = generate_sequences(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

    # Score each sequence
    print(f"\nScoring with motif alignment...\n")

    total_components = 0
    total_found = 0
    category_stats = defaultdict(lambda: {"expected": 0, "found": 0, "scores": []})
    all_rewards = []
    all_details = []

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        seq = extract_dna(completion)
        has_eos = "<EOS>" in completion or "</s>" in completion

        details = scorer.score_sequence_detailed(prompt, seq)
        reward = details["reward"]
        all_rewards.append(reward)
        all_details.append(details)

        hard_tokens = parse_hard_tokens(prompt, lookup_df)
        total_components += len(hard_tokens)
        total_found += details["n_found"]

        # Per-category tracking
        for motif in details["per_motif"]:
            token = motif["token"]
            inner = token.strip("<>")
            cat = inner.split("_")[0] if "_" in inner else "OTHER"
            category_stats[cat]["expected"] += 1
            if motif["found"]:
                category_stats[cat]["found"] += 1
            category_stats[cat]["scores"].append(motif["score_ratio"])

        if i < 10 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] tokens={len(hard_tokens)} found={details['n_found']} "
                  f"reward={reward:.2f} seq_len={len(seq)} eos={has_eos}")
            # Show per-component breakdown for first few
            if i < 5:
                for motif, cscore in zip(details["per_motif"], details["component_scores"]):
                    status = "OK" if motif["found"] else "  "
                    print(f"    {status} {motif['token']:30s} score_ratio={motif['score_ratio']:.3f} "
                          f"component_score={cscore:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    import numpy as np
    rewards = np.array(all_rewards)
    print(f"\nReward:  mean={rewards.mean():.3f}  std={rewards.std():.3f}  "
          f"min={rewards.min():.3f}  max={rewards.max():.3f}")
    print(f"Components: {total_found}/{total_components} found "
          f"({100*total_found/max(total_components,1):.1f}%)")

    n_components_per_prompt = [d["n_hard_tokens"] for d in all_details]
    print(f"Components per prompt: mean={np.mean(n_components_per_prompt):.1f}  "
          f"range=[{min(n_components_per_prompt)}, {max(n_components_per_prompt)}]")

    print(f"\nPer-category breakdown:")
    print(f"  {'Category':12s} {'Found':>6s} {'Expected':>9s} {'Rate':>6s} {'Avg Score':>10s}")
    print(f"  {'-'*12} {'-'*6} {'-'*9} {'-'*6} {'-'*10}")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        rate = s["found"] / max(s["expected"], 1)
        avg_score = np.mean(s["scores"]) if s["scores"] else 0
        print(f"  {cat:12s} {s['found']:6d} {s['expected']:9d} {rate:5.1%} {avg_score:10.3f}")

    print(f"\n  {'TOTAL':12s} {total_found:6d} {total_components:9d} "
          f"{total_found/max(total_components,1):5.1%}")


if __name__ == "__main__":
    main()
