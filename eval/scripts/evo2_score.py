#!/usr/bin/env python3
"""Score plasmid sequences with Evo2: perplexity + embeddings.

Loads matched real/generated pairs and computes per-sequence:
  - Perplexity (from cross-entropy loss)
  - Mean-pooled hidden state embeddings

Sequences longer than max_len are scored on a sliding window and averaged.

Usage:
    python eval/scripts/evo2_score.py \
        --pairs eval/runs/grpo_plannotate_full_20260408/matched_pairs.parquet \
        --output-dir eval/runs/grpo_plannotate_full_20260408/evo2 \
        --model evo2_7b_base \
        --max-len 8000 \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def score_sequence(model, seq: str, max_len: int = 8000, device: str = "cuda") -> dict:
    """Score a DNA sequence: perplexity + embedding from Evo2.

    For sequences > max_len, uses sliding windows with 50% overlap and averages.
    """
    if len(seq) < 10:
        return {"perplexity": float("nan"), "embedding": None}

    tokenizer = model.tokenizer

    if len(seq) <= max_len:
        windows = [seq]
    else:
        # Sliding windows with 50% overlap
        step = max_len // 2
        windows = []
        for start in range(0, len(seq) - max_len // 2, step):
            windows.append(seq[start:start + max_len])
        if not windows:
            windows = [seq[:max_len]]

    all_losses = []
    all_embeddings = []

    for window in windows:
        input_ids = torch.tensor(
            tokenizer.tokenize(window),
            dtype=torch.int,
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs, embeddings = model(input_ids)

        logits = outputs[0]  # (batch, seq_len, vocab)

        # Compute per-token cross-entropy loss
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].long().contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        all_losses.append(loss.item())

        # Mean-pool the last hidden state for embedding
        if embeddings is not None:
            # embeddings shape depends on model - typically (batch, seq_len, hidden_dim)
            emb = embeddings[0].mean(dim=0).cpu().numpy()  # (hidden_dim,)
            all_embeddings.append(emb)

    avg_loss = np.mean(all_losses)
    perplexity = np.exp(avg_loss)

    if all_embeddings:
        avg_embedding = np.mean(all_embeddings, axis=0)
    else:
        avg_embedding = None

    return {"perplexity": float(perplexity), "embedding": avg_embedding}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="matched_pairs.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="evo2_7b_base")
    parser.add_argument("--max-len", type=int, default=8000,
                        help="Max sequence length per forward pass (tokens)")
    parser.add_argument("--n", type=int, default=None,
                        help="Subsample N pairs (default: all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(args.pairs)
    if args.n:
        pairs = pairs.sample(n=min(args.n, len(pairs)), random_state=42).reset_index(drop=True)
    print(f"Scoring {len(pairs)} pairs with {args.model}")

    # Load Evo2
    print(f"Loading {args.model}...")
    t0 = time.time()
    from evo2 import Evo2
    model = Evo2(args.model)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Score all pairs
    results = []
    gen_embeddings = []
    real_embeddings = []

    for idx, row in pairs.iterrows():
        t0 = time.time()

        # Score generated
        gen_result = score_sequence(model, row["gen_sequence"], max_len=args.max_len, device=args.device)

        # Score real
        real_result = score_sequence(model, row["real_sequence"], max_len=args.max_len, device=args.device)

        results.append({
            "gen_id": row["gen_id"],
            "prompt": row["prompt"],
            "real_plasmid_id": row["real_plasmid_id"],
            "gen_length": row["gen_length"],
            "real_length": row["real_length"],
            "gen_perplexity": gen_result["perplexity"],
            "real_perplexity": real_result["perplexity"],
            "ppl_ratio": gen_result["perplexity"] / max(real_result["perplexity"], 1e-10),
        })

        if gen_result["embedding"] is not None:
            gen_embeddings.append(gen_result["embedding"])
        if real_result["embedding"] is not None:
            real_embeddings.append(real_result["embedding"])

        elapsed = time.time() - t0
        if (idx + 1) % 10 == 0 or idx == 0:
            gen_ppl = gen_result["perplexity"]
            real_ppl = real_result["perplexity"]
            print(f"  [{idx+1}/{len(pairs)}] gen_ppl={gen_ppl:.1f} real_ppl={real_ppl:.1f} "
                  f"ratio={gen_ppl/max(real_ppl,1e-10):.2f} ({elapsed:.1f}s/pair)", flush=True)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_parquet(out / "evo2_scores.parquet")

    # Save embeddings as numpy arrays
    if gen_embeddings:
        gen_emb_arr = np.stack(gen_embeddings)
        real_emb_arr = np.stack(real_embeddings)
        np.save(out / "gen_embeddings.npy", gen_emb_arr)
        np.save(out / "real_embeddings.npy", real_emb_arr)
        print(f"Embeddings: gen={gen_emb_arr.shape}, real={real_emb_arr.shape}")

    # Summary
    valid = results_df.dropna(subset=["gen_perplexity", "real_perplexity"])
    summary = {
        "model": args.model,
        "n_scored": len(valid),
        "gen_perplexity_mean": float(valid["gen_perplexity"].mean()),
        "gen_perplexity_median": float(valid["gen_perplexity"].median()),
        "real_perplexity_mean": float(valid["real_perplexity"].mean()),
        "real_perplexity_median": float(valid["real_perplexity"].median()),
        "ppl_ratio_mean": float(valid["ppl_ratio"].mean()),
        "ppl_ratio_median": float(valid["ppl_ratio"].median()),
        "embedding_dim": gen_emb_arr.shape[1] if gen_embeddings else 0,
    }
    with open(out / "evo2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evo2 Scoring Summary ({args.model})")
    print(f"{'='*60}")
    print(f"  Generated perplexity: {summary['gen_perplexity_mean']:.1f} (mean), {summary['gen_perplexity_median']:.1f} (median)")
    print(f"  Real perplexity:      {summary['real_perplexity_mean']:.1f} (mean), {summary['real_perplexity_median']:.1f} (median)")
    print(f"  PPL ratio (gen/real): {summary['ppl_ratio_mean']:.2f} (mean), {summary['ppl_ratio_median']:.2f} (median)")
    print(f"  Embeddings: {summary['embedding_dim']}D")
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
