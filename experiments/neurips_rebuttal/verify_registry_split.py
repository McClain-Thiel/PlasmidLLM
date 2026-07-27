#!/usr/bin/env python3
"""Check within-token reward/held-out registry separation with exhaustive BLAST."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def write_fasta(rows: pd.DataFrame, path: Path, prefix: str) -> dict[str, str]:
    token_by_id = {}
    with path.open("w") as handle:
        for i, row in rows.reset_index(drop=True).iterrows():
            seq_id = f"{prefix}{i}"
            token_by_id[seq_id] = row["token"]
            handle.write(f">{seq_id}\n{row['sequence']}\n")
    return token_by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    reward = pd.read_parquet(args.reward)
    heldout = pd.read_parquet(args.heldout)
    reward = reward[reward["sequence"].notna()].copy()
    heldout = heldout[heldout["sequence"].notna()].copy()

    with tempfile.TemporaryDirectory(prefix="registry_split_") as raw_tmp:
        tmp = Path(raw_tmp)
        reward_fasta = tmp / "reward.fasta"
        heldout_fasta = tmp / "heldout.fasta"
        reward_tokens = write_fasta(reward, reward_fasta, "r")
        heldout_tokens = write_fasta(heldout, heldout_fasta, "h")
        database = tmp / "reward_db"
        output = tmp / "hits.tsv"

        subprocess.run(
            [
                "makeblastdb",
                "-in",
                str(reward_fasta),
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
                str(heldout_fasta),
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
                str(len(reward)),
                "-num_threads",
                str(args.threads),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        columns = "qseqid sseqid pident length qlen slen bitscore".split()
        hits = pd.read_csv(output, sep="\t", names=columns)
        hits["query_token"] = hits["qseqid"].map(heldout_tokens)
        hits["subject_token"] = hits["sseqid"].map(reward_tokens)
        same_token = hits[hits["query_token"] == hits["subject_token"]].copy()
        same_token["shorter_coverage"] = same_token["length"] / same_token[["qlen", "slen"]].min(
            axis=1
        )
        leaks = same_token[
            (same_token["pident"] >= 90) & (same_token["shorter_coverage"] >= 0.8)
        ].sort_values(["pident", "shorter_coverage"], ascending=False)

        print(
            f"reward_sequences={len(reward)} heldout_sequences={len(heldout)} "
            f"same_token_hsps={len(same_token)} leaks={len(leaks)}"
        )
        if len(same_token):
            top = same_token.sort_values(["shorter_coverage", "pident"], ascending=False).head(10)
            print(
                top[
                    [
                        "qseqid",
                        "sseqid",
                        "query_token",
                        "pident",
                        "shorter_coverage",
                        "qlen",
                        "slen",
                    ]
                ].to_string(index=False)
            )
        if len(leaks):
            raise RuntimeError(
                "Registry split contains same-token pairs at >=90% identity "
                "over >=80% of the shorter sequence"
            )


if __name__ == "__main__":
    main()
