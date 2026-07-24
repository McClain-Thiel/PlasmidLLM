#!/usr/bin/env python3
"""Quantify how much of each generated plasmid is explained by its single most
similar training plasmid, with real held-out Addgene plasmids as the control.

Written for the NeurIPS 2026 rebuttal (meta-review point 2; Dok5 W2/Q1; pSXh W4).
sourmash containment — what the paper currently reports — does not resolve
near-duplicate or sub-plasmid reuse, which is the specific objection raised. This
script does: BLASTN each query against the training corpus, take the single best
subject by union-of-HSP query coverage, and report the coverage/identity joint
distribution.

The control matters more than the treatment here. Generated sequences will always
share long stretches with training plasmids because plasmids are built from a
shared parts catalogue, so a raw coverage number is uninterpretable on its own.
Real Addgene plasmids that are *not* in the training corpus give the reference
point for how similar a genuinely novel plasmid looks.

Usage:
  python eval/scripts/train_similarity_blast.py \\
      --train-pairs training_pairs_v4.parquet \\
      --generations runs/grpo_plannotate_full_20260408/generations.parquet \\
      --control addgene_500.fasta \\
      --tier3 runs/grpo_plannotate_full_20260408/metrics/tier3_essentials.parquet \\
      --workdir /tmp/trainsim
"""

from __future__ import annotations

import argparse
import collections
import statistics
import subprocess
from pathlib import Path

import duckdb


def union_coverage(intervals: list[tuple[int, int, float]]) -> int:
    """Total query bases covered by the union of HSPs.

    HSPs overlap and repeat, so summing hit lengths double-counts badly — a
    tandem-repeat generation can sum past its own length.
    """
    spans = sorted((min(a, b), max(a, b)) for a, b, _ in intervals)
    total = 0
    cur_start, cur_end = spans[0]
    for a, b in spans[1:]:
        if a <= cur_end + 1:
            cur_end = max(cur_end, b)
        else:
            total += cur_end - cur_start + 1
            cur_start, cur_end = a, b
    return total + cur_end - cur_start + 1


def best_match_per_query(tsv: Path) -> dict[str, tuple[float, float, str]]:
    """query -> (coverage_fraction, length_weighted_identity, best_subject)."""
    hsps: dict[tuple[str, str], list] = collections.defaultdict(list)
    qlen: dict[str, int] = {}
    with open(tsv) as fh:
        for line in fh:
            q, s, pident, _aln, ql, _sl, qstart, qend, _bits = line.rstrip("\n").split("\t")
            qlen[q] = int(ql)
            hsps[(q, s)].append((int(qstart), int(qend), float(pident)))

    best: dict[str, tuple[float, float, str]] = {}
    for (q, s), ivs in hsps.items():
        cov = union_coverage(ivs) / qlen[q]
        weight = sum(abs(b - a) + 1 for a, b, _ in ivs)
        ident = sum(p * (abs(b - a) + 1) for a, b, p in ivs) / weight
        if q not in best or cov > best[q][0]:
            best[q] = (cov, ident, s)
    return best


def write_fasta(rows, path: Path) -> int:
    n = 0
    with open(path, "w") as fh:
        for name, seq in rows:
            seq = (seq or "").strip()
            if seq:
                fh.write(f">{name}\n{seq}\n")
                n += 1
    return n


def blast(query: Path, db: Path, out: Path, threads: int) -> None:
    subprocess.run(
        ["blastn", "-query", str(query), "-db", str(db), "-out", str(out),
         "-outfmt", "6 qseqid sseqid pident length qlen slen qstart qend bitscore",
         "-max_target_seqs", "5", "-evalue", "1e-10", "-num_threads", str(threads)],
        check=True,
    )


def summarise(name: str, best: dict, total: int) -> dict:
    covs = [best[q][0] if q in best else 0.0 for q in best] or [0.0]
    near = sum(1 for c, i, _ in best.values() if c >= 0.90 and i >= 99)
    deriv = sum(1 for c, i, _ in best.values() if c >= 0.75 and i >= 95)
    return {
        "panel": name, "n": total, "no_hit": total - len(best),
        "median_cov": statistics.median(covs),
        "near_dup_pct": near / total * 100,
        "derivative_pct": deriv / total * 100,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-pairs", type=Path, required=True)
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True,
                    help="FASTA of real plasmids, headers '<plasmid_id>|<name>'")
    ap.add_argument("--tier3", type=Path, default=None,
                    help="tier3_essentials.parquet; adds a viable-only breakdown")
    ap.add_argument("--workdir", type=Path, default=Path("trainsim"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    w = args.workdir
    w.mkdir(parents=True, exist_ok=True)

    train_rows = duckdb.sql(
        f"select plasmid_id, any_value(sequence) from '{args.train_pairs}' group by 1"
    ).fetchall()
    n_train = write_fasta(train_rows, w / "train.fasta")
    train_ids = {str(r[0]) for r in train_rows}
    print(f"training plasmids: {n_train}")

    subprocess.run(["makeblastdb", "-in", str(w / "train.fasta"), "-dbtype", "nucl",
                    "-out", str(w / "traindb")], check=True, capture_output=True)

    gen_rows = duckdb.sql(f"select sequence from '{args.generations}'").fetchall()
    n_gen = write_fasta(((f"gen_{i}", r[0]) for i, r in enumerate(gen_rows)), w / "gen.fasta")
    print(f"generations: {n_gen}")

    blast(w / "gen.fasta", w / "traindb", w / "gen_vs_train.tsv", args.threads)
    blast(args.control, w / "traindb", w / "real_vs_train.tsv", args.threads)

    gen_best = best_match_per_query(w / "gen_vs_train.tsv")
    real_best = best_match_per_query(w / "real_vs_train.tsv")

    # Drop control entries that are themselves in the training corpus, or the
    # control measures self-matches rather than novelty.
    real_best = {q: v for q, v in real_best.items() if q.split("|")[0] not in train_ids}
    n_control = sum(
        1 for line in open(args.control)
        if line.startswith(">") and line[1:].split("|")[0] not in train_ids
    )

    rows = [summarise("generated", gen_best, n_gen),
            summarise("real held-out Addgene", real_best, n_control)]

    if args.tier3:
        viable = {
            f"gen_{r[0].split('_')[1]}"
            for r in duckdb.sql(
                f"select seq_id, plausibility_pass from '{args.tier3}'").fetchall()
            if r[1]
        }
        rows.insert(1, summarise("generated (viable only)",
                                 {q: v for q, v in gen_best.items() if q in viable},
                                 len(viable)))

    hdr = f"{'panel':<28}{'n':>6}{'no hit':>8}{'med cov':>10}{'near-dup':>10}{'derivative':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['panel']:<28}{r['n']:>6}{r['no_hit']:>8}{r['median_cov']:>10.3f}"
              f"{r['near_dup_pct']:>9.1f}%{r['derivative_pct']:>11.1f}%")
    print("\nnear-dup = best-match coverage >=90% and identity >=99%")
    print("derivative = coverage >=75% and identity >=95%")


if __name__ == "__main__":
    main()
