#!/usr/bin/env python3
"""Build negative control baselines for the eval suite.

Generates three baseline sets from the Addgene-500 reference panel:
1. Random sequences — length-matched, GC-matched random DNA (floor)
2. Shuffled Addgene — dinucleotide-shuffled versions (tests architecture metrics)
3. Held-out real Addgene — the reference panel itself (ceiling)

Outputs FASTA files into eval/baselines/{random,shuffled,real}/.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


def gc_matched_random(length: int, gc: float, rng: np.random.Generator) -> str:
    """Generate random DNA with specified length and GC content."""
    n_gc = int(length * gc)
    n_at = length - n_gc
    bases = list("G" * (n_gc // 2) + "C" * (n_gc - n_gc // 2) +
                 "A" * (n_at // 2) + "T" * (n_at - n_at // 2))
    rng.shuffle(bases)
    return "".join(bases)


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Dinucleotide-preserving shuffle using Altschul-Erikson algorithm.

    Preserves exact dinucleotide frequencies, which controls for local
    composition while destroying higher-order structure.
    """
    seq = seq.upper()
    if len(seq) < 3:
        return seq

    # Build Eulerian graph of dinucleotides
    from collections import defaultdict
    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])

    # Shuffle edges (except last edge from each node to preserve Eulerian path)
    for base in edges:
        edge_list = edges[base]
        if len(edge_list) > 1:
            # Keep last edge fixed, shuffle the rest
            last = edge_list[-1]
            rest = edge_list[:-1]
            rng.shuffle(rest)
            edges[base] = list(rest) + [last]

    # Walk the Eulerian path
    result = [seq[0]]
    current = seq[0]
    edge_idx = defaultdict(int)
    for _ in range(len(seq) - 1):
        idx = edge_idx[current]
        if idx < len(edges[current]):
            next_base = edges[current][idx]
            edge_idx[current] += 1
            result.append(next_base)
            current = next_base
        else:
            break

    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="eval/reference/addgene_reference_500.csv")
    parser.add_argument("--output-dir", default="eval/baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)

    # 1. Random sequences
    random_dir = out / "random"
    random_dir.mkdir(parents=True, exist_ok=True)
    with open(random_dir / "sequences.fasta", "w") as f:
        for _, row in ref.iterrows():
            seq = row["sequence"]
            gc = (seq.upper().count("G") + seq.upper().count("C")) / len(seq)
            rand_seq = gc_matched_random(len(seq), gc, rng)
            f.write(f">random_{row['id']}|len={len(seq)}|gc={gc:.3f}\n{rand_seq}\n")
    print(f"Random: {len(ref)} sequences -> {random_dir / 'sequences.fasta'}")

    # 2. Dinucleotide-shuffled
    shuffled_dir = out / "shuffled"
    shuffled_dir.mkdir(parents=True, exist_ok=True)
    with open(shuffled_dir / "sequences.fasta", "w") as f:
        for _, row in ref.iterrows():
            shuf_seq = dinucleotide_shuffle(row["sequence"], rng)
            f.write(f">shuffled_{row['id']}|{row['name']}\n{shuf_seq}\n")
    print(f"Shuffled: {len(ref)} sequences -> {shuffled_dir / 'sequences.fasta'}")

    # 3. Real held-out (just copy the reference)
    real_dir = out / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    with open(real_dir / "sequences.fasta", "w") as f:
        for _, row in ref.iterrows():
            f.write(f">{row['id']}|{row['name']}\n{row['sequence']}\n")
    print(f"Real: {len(ref)} sequences -> {real_dir / 'sequences.fasta'}")


if __name__ == "__main__":
    main()
