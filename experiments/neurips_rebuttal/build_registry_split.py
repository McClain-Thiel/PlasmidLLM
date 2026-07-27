#!/usr/bin/env python3
"""Build a token-stratified registry split with no within-token cross-arm near duplicates.

Rows are connected when they map to the same prompt token and share a BLAST HSP at
>=90% identity over >=80% of the shorter reference. Whole connected components are then
assigned to one arm. This isolates exact reward-reference reuse for a requested token; it
does not claim that the underlying sequences were absent from pretraining.
"""

import argparse
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


class UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def closest_nontrivial_subset(components: list[list[int]], target: float) -> set[int]:
    """Select whole components whose row count is closest to target."""
    possible: dict[int, tuple[int, ...]] = {0: ()}
    for component_index, component in enumerate(components):
        size = len(component)
        additions = {
            total + size: chosen + (component_index,) for total, chosen in possible.items()
        }
        for total, chosen in additions.items():
            possible.setdefault(total, chosen)

    total_rows = sum(map(len, components))
    candidates = [
        (abs(total - target), total, chosen)
        for total, chosen in possible.items()
        if 0 < total < total_rows
    ]
    if not candidates:
        return set()
    _, _, chosen = min(candidates)
    return set(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reward-output", type=Path)
    parser.add_argument("--heldout-output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    registry = pd.read_parquet(args.registry).reset_index(drop=True)
    valid = registry[registry["sequence"].notna() & (registry["sequence"].str.len() >= 10)].copy()
    valid_indices = valid.index.tolist()
    union_find = UnionFind(valid_indices)

    with tempfile.TemporaryDirectory(prefix="registry_split_v2_") as raw_tmp:
        tmp = Path(raw_tmp)
        fasta = tmp / "registry.fasta"
        database = tmp / "registry_db"
        output = tmp / "hits.tsv"
        with fasta.open("w") as handle:
            for index, row in valid.iterrows():
                handle.write(f">row{index}\n{row['sequence']}\n")

        subprocess.run(
            [
                "makeblastdb",
                "-in",
                str(fasta),
                "-dbtype",
                "nucl",
                "-out",
                str(database),
                "-parse_seqids",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "blastn",
                "-query",
                str(fasta),
                "-db",
                str(database),
                "-out",
                str(output),
                "-outfmt",
                "6 qseqid sseqid pident length qlen slen bitscore",
                "-evalue",
                "10",
                "-word_size",
                "7",
                "-dust",
                "no",
                "-soft_masking",
                "false",
                "-max_target_seqs",
                str(len(valid)),
                "-num_threads",
                str(args.threads),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        columns = "qseqid sseqid pident length qlen slen bitscore".split()
        hits = pd.read_csv(output, sep="\t", names=columns)
        hits["query_index"] = hits["qseqid"].str.removeprefix("row").astype(int)
        hits["subject_index"] = hits["sseqid"].str.removeprefix("row").astype(int)
        hits["query_token"] = hits["query_index"].map(registry["token"])
        hits["subject_token"] = hits["subject_index"].map(registry["token"])
        hits["shorter_coverage"] = hits["length"] / hits[["qlen", "slen"]].min(axis=1)
        edges = hits[
            (hits["query_index"] != hits["subject_index"])
            & (hits["query_token"] == hits["subject_token"])
            & (hits["pident"] >= 90)
            & (hits["shorter_coverage"] >= 0.8)
        ]
        for row in edges.itertuples():
            union_find.union(row.query_index, row.subject_index)

    by_token_and_component: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in valid.iterrows():
        by_token_and_component[row["token"]][union_find.find(index)].append(index)

    heldout_indices = set()
    split_token_stats = []
    for token, component_map in sorted(by_token_and_component.items()):
        components = sorted(
            component_map.values(),
            key=lambda values: (len(values), min(values)),
            reverse=True,
        )
        chosen = closest_nontrivial_subset(components, len(valid.loc[valid["token"] == token]) / 2)
        for component_index in chosen:
            heldout_indices.update(components[component_index])
        split_token_stats.append(
            {
                "token": token,
                "components": len(components),
                "reward": sum(
                    index not in heldout_indices for component in components for index in component
                ),
                "heldout": sum(
                    index in heldout_indices for component in components for index in component
                ),
            }
        )

    reward = registry.loc[~registry.index.isin(heldout_indices)].copy()
    heldout = registry.loc[registry.index.isin(heldout_indices)].copy()
    reward_tokens = set(reward["token"])
    heldout_tokens = set(heldout["token"])
    if not heldout_tokens <= reward_tokens:
        raise RuntimeError("Every held-out token must retain at least one reward reference")

    cross_edges = edges[
        edges["query_index"].isin(heldout_indices) & ~edges["subject_index"].isin(heldout_indices)
    ]
    if len(cross_edges):
        raise RuntimeError(f"Found {len(cross_edges)} cross-split similarity edges")

    print(
        f"registry_rows={len(registry)} valid_sequences={len(valid)} similarity_edges={len(edges)}"
    )
    print(
        f"reward_rows={len(reward)} reward_tokens={reward.token.nunique()} "
        f"heldout_rows={len(heldout)} heldout_tokens={heldout.token.nunique()}"
    )
    stats = pd.DataFrame(split_token_stats)
    print(stats[stats["heldout"] > 0].to_string(index=False))

    if args.reward_output:
        args.reward_output.parent.mkdir(parents=True, exist_ok=True)
        reward.to_parquet(args.reward_output, index=False)
    if args.heldout_output:
        args.heldout_output.parent.mkdir(parents=True, exist_ok=True)
        heldout.to_parquet(args.heldout_output, index=False)


if __name__ == "__main__":
    main()
