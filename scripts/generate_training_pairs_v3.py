#!/usr/bin/env python3
"""Generate v3 training pairs using plannotate feature annotations as conditioning tokens.

Improvements over v2:
  - Conditioning tokens derived from actual plannotate annotations (not metadata buckets)
  - EOS-safe: only includes sequences where full BOS+prompt+SEP+DNA+EOS fits in max_seq_len
  - Same random-subset uniqueness approach as v2 (shuffled feature subsets)

Output: training_pairs_v3.parquet with columns:
  plasmid_id, token_prompt, sequence, sequence_length, num_tokens
"""

import argparse
import random
import re
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

BASE = "/mnt/s3/phd-research-storage-1758274488/addgene_clean"

# Quality thresholds for plannotate annotations
MIN_PERCMATCH = 95.0
EXCLUDE_FRAGMENTS = True

# How many top features to include in the vocabulary
TOP_N_FEATURES = 150

# Random-subset parameters (same as v2)
MIN_TOKENS = 3
MAX_ATTEMPTS = 500
SEED = 42

# EOS safety: BOS(1) + prompt_tokens + SEP(1) + dna_chars + EOS(1) <= max_seq_len
# Using a conservative budget: reserve 20 tokens for prompt (adjustable)
DEFAULT_MAX_SEQ_LEN = 8192


def feature_to_token(feature: str) -> str:
    """Convert a plannotate feature name to a valid <TOKEN> string."""
    # Uppercase, replace spaces/special chars with underscores
    tok = feature.upper()
    tok = re.sub(r"[^A-Z0-9]+", "_", tok)
    tok = tok.strip("_")
    # Collapse repeated underscores
    tok = re.sub(r"_+", "_", tok)
    return f"<FEAT_{tok}>"


def fits_in_context(n_prompt_tokens: int, seq_len: int, max_seq_len: int) -> bool:
    """Check if BOS + prompt + SEP + DNA + EOS fits in max_seq_len."""
    return (1 + n_prompt_tokens + 1 + seq_len + 1) <= max_seq_len


def find_unique_prompt(tokens: list, used: set, min_tokens: int, max_attempts: int):
    n = len(tokens)
    if n < min_tokens:
        return None
    for _ in range(max_attempts):
        size = random.randint(min_tokens, n)
        subset = random.sample(tokens, size)
        prompt = " ".join(subset)
        if prompt not in used:
            return prompt
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pairs",
                        default=f"{BASE}/tokenization/training_pairs.parquet")
    parser.add_argument("--annotations",
                        default=f"{BASE}/annotations/plannotate_annotations.parquet")
    parser.add_argument("--output",
                        default=f"{BASE}/tokenization/training_pairs_v3.parquet")
    parser.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--min-percmatch", type=float, default=MIN_PERCMATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--vocab-out", default=f"{BASE}/tokenization/token_vocabulary_v3.json",
                        help="Path to write updated vocab JSON")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load training sequences ───────────────────────────────────────────────
    print("Loading training pairs...")
    tp = pq.read_table(args.training_pairs,
                       columns=["plasmid_id", "sequence", "sequence_length"]).to_pandas()
    tp["plasmid_id"] = tp["plasmid_id"].astype(str)
    print(f"  {len(tp):,} sequences, {tp['plasmid_id'].nunique():,} unique plasmids")

    # ── Load plannotate annotations ───────────────────────────────────────────
    print("Loading plannotate annotations...")
    ann = pq.read_table(args.annotations,
                        columns=["plasmid_id", "Feature", "percmatch", "fragment"]).to_pandas()
    ann["plasmid_id"] = ann["plasmid_id"].astype(str)

    # Quality filter
    ann = ann[ann["percmatch"] >= args.min_percmatch]
    if EXCLUDE_FRAGMENTS:
        ann = ann[~ann["fragment"].fillna(False).astype(bool)]
    print(f"  {len(ann):,} annotations after quality filter")
    print(f"  {ann['plasmid_id'].nunique():,} plasmids with annotations")

    # ── Select top-N features ─────────────────────────────────────────────────
    feat_counts = ann.groupby("Feature")["plasmid_id"].nunique().sort_values(ascending=False)
    top_features = set(feat_counts.head(args.top_n_features).index)
    print(f"\nTop {args.top_n_features} features selected:")
    for feat in sorted(feat_counts.head(args.top_n_features).index):
        tok = feature_to_token(feat)
        cnt = feat_counts[feat]
        print(f"  {cnt:>7,}  {feat:<40}  {tok}")

    # Build feature → token mapping
    feat_to_tok = {f: feature_to_token(f) for f in top_features}
    all_new_tokens = sorted(set(feat_to_tok.values()))
    print(f"\n{len(all_new_tokens)} unique feature tokens")

    # ── Build per-plasmid feature token list ─────────────────────────────────
    ann_top = ann[ann["Feature"].isin(top_features)].copy()
    ann_top["token"] = ann_top["Feature"].map(feat_to_tok)

    plasmid_tokens = (
        ann_top.groupby("plasmid_id")["token"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    # ── EOS-safe filter ───────────────────────────────────────────────────────
    print(f"\nApplying EOS-safe filter (max_seq_len={args.max_seq_len})...")

    # For EOS check, we use a conservative estimate of prompt token count.
    # Actual prompt varies by subset, so we use the FULL feature set length as upper bound.
    # At generation time, the model will use a subset (so it always fits).
    # But we need to ensure at least MIN_TOKENS of features + BOS + SEP + EOS + DNA fits.
    # Conservative: reserve max(full_feature_count, min_tokens) + 2 for BOS+SEP.

    # Actually: filter based on whether the SMALLEST valid prompt (min_tokens features)
    # leaves room for the DNA sequence.
    def can_fit(row):
        pid = row["plasmid_id"]
        feats = plasmid_tokens.get(pid, [])
        if len(feats) < args.min_tokens:
            return False
        # Best case: smallest prompt = min_tokens features (each ~15 chars avg)
        # Worst case: max features. We filter on DNA fitting with max prompt.
        # Actually use min_tokens as the prompt we'll generate for this row.
        return fits_in_context(len(feats), int(row["sequence_length"]), args.max_seq_len)

    tp["has_annotations"] = tp["plasmid_id"].isin(plasmid_tokens)
    tp_ann = tp[tp["has_annotations"]].copy()
    print(f"  With annotations: {len(tp_ann):,}/{len(tp):,}")

    tp_ann["fits"] = tp_ann.apply(can_fit, axis=1)
    tp_fit = tp_ann[tp_ann["fits"]].copy()
    print(f"  EOS-safe (max prompt + DNA <= {args.max_seq_len}): {len(tp_fit):,} ({100*len(tp_fit)/len(tp):.1f}% of all)")
    print(f"  Dropped (too long): {len(tp_ann)-len(tp_fit):,}")

    # ── Assign unique prompts (same algorithm as v2) ──────────────────────────
    print("\nAssigning unique prompts via random subset approach...")

    used_prompts = set()
    results = []
    dropped = 0

    for _, row in tp_fit.iterrows():
        pid = row["plasmid_id"]
        tokens = plasmid_tokens[pid]  # sorted list of feature tokens for this plasmid

        prompt = find_unique_prompt(tokens, used_prompts, args.min_tokens, args.max_attempts)
        if prompt is None:
            dropped += 1
            continue

        used_prompts.add(prompt)
        n_tok = len(prompt.split())
        results.append({
            "plasmid_id": pid,
            "token_prompt": prompt,
            "num_tokens": n_tok,
            "sequence": row["sequence"],
            "sequence_length": int(row["sequence_length"]),
        })

    print(f"  Dropped (exhausted unique search): {dropped:,}")
    result_df = pd.DataFrame(results)
    n_unique = result_df["token_prompt"].nunique()
    print(f"\nOutput: {len(result_df):,} rows, {n_unique:,} unique prompts")
    assert n_unique == len(result_df), "BUG: duplicate prompts!"

    print(f"\nToken count distribution:")
    print(result_df["num_tokens"].describe().to_string())
    print(f"\nSequence length distribution (bp):")
    print(result_df["sequence_length"].describe().to_string())

    # ── Save dataset ──────────────────────────────────────────────────────────
    print(f"\nSaving to {args.output}...")
    result_df.to_parquet(args.output, index=False)
    print("Done.")

    # ── Build and save updated vocabulary ────────────────────────────────────
    import json
    from pathlib import Path

    # Load existing vocab
    existing_vocab_path = f"{BASE}/tokenization/token_vocabulary.json"
    with open(existing_vocab_path) as f:
        vocab_data = json.load(f)
    if isinstance(vocab_data, dict) and "token_to_id" in vocab_data:
        vocab = vocab_data["token_to_id"]
    else:
        vocab = vocab_data

    # Add new feature tokens
    next_id = max(vocab.values()) + 1
    added = 0
    for tok in all_new_tokens:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
            added += 1

    print(f"\nVocab: added {added} new tokens, total {len(vocab)}")
    Path(args.vocab_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.vocab_out, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to {args.vocab_out}")

    # Print summary of new tokens
    print("\nNew feature tokens added to vocab:")
    for tok in all_new_tokens:
        print(f"  {tok}")


if __name__ == "__main__":
    main()
