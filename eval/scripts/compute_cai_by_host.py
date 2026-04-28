#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow>=14.0",
#     "numpy>=1.26",
#     "python_codon_tables>=0.1.12",
# ]
# ///
"""Compute per-ORF Codon Adaptation Index (Sharp & Li 1987) for generated plasmids
against three host codon-usage tables, conditioned on the prompt's SP_ token.

Inputs (under --cache, flat bucket layout):
    <run>/generations.parquet     — sequence per prompt, prompt column
    <run>/orfs/prodigal.gff       — ORF coordinates (Prodigal v2.6.3, table 11)

Outputs (under --out):
    cai_per_orf_<run>.parquet     — one row per ORF kept (length ≥ MIN_CODONS,
                                     non-partial, no internal stops)
    cai_per_seq_<run>.parquet     — aggregated per generation: mean/length-weighted
                                     CAI per host, longest-ORF CAI per host, n_orfs

Codon-usage tables come from `python_codon_tables` which sources from the Kazusa
Codon Usage Database (Nakamura et al., NAR 2000). These are whole-CDS frequencies,
not Sharp & Li's HEG set — adequate for cross-host comparison but we should add
a HEG sensitivity check before publication.

Method (Sharp & Li, NAR 1987):
    For each amino acid family with > 1 codon, define
        w(c) = freq(c) / max_{c' ∈ syn(c)} freq(c')
    For an ORF of N codons (excluding the start, single-codon families {ATG, TGG},
    and stop codons), define
        CAI = exp( (1/N') sum log w(c_i) )
    where N' is the number of contributing codons and only codons with non-zero
    reference frequency contribute (zero-frequency codons treated per Sharp & Li
    by adding a pseudocount of 0.5 to the table prior to normalisation).

Usage:
    uv run --script eval/scripts/compute_cai_by_host.py \\
        --cache .paper-audit/bucket \\
        --runs grpo_plannotate_full_20260408 base_kmer6_matched_20260420_1303 \\
        --out eval/figures/paper/cai
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import python_codon_tables as pct


HOSTS = {
    "ecoli": "e_coli_316407",
    "yeast": "s_cerevisiae_4932",
    "human": "h_sapiens_9606",
}

# Single-codon AAs (Met, Trp) and stop codons are excluded from CAI
SINGLE_CODON_AAS = {"M", "W"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
PSEUDOCOUNT = 0.5  # Sharp & Li recommend non-zero w; common practice
MIN_CODONS = 100   # filter short ORFs (Sharp & Li 1987)
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANTGCAn")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def build_w_table(host_table: dict[str, dict[str, float]]) -> dict[str, float]:
    """Convert a Kazusa-style {aa: {codon: fraction_in_aa}} to {codon: w(c)}.

    Pseudocount is applied to zero entries before normalising so log(w) is
    finite. Single-codon families (M, W) and stop codons get w=1.0 by
    convention but never contribute to CAI (excluded at the call site).
    """
    w: dict[str, float] = {}
    for aa, codons in host_table.items():
        if aa == "*":
            for c in codons:
                w[c] = 1.0
            continue
        if len(codons) == 1:
            for c in codons:
                w[c] = 1.0
            continue
        # Apply pseudocount, renormalise within aa, build w
        adjusted = {c: f + PSEUDOCOUNT / 1000.0 for c, f in codons.items()}
        total = sum(adjusted.values())
        adjusted = {c: f / total for c, f in adjusted.items()}
        max_f = max(adjusted.values())
        for c, f in adjusted.items():
            w[c] = f / max_f
    return w


def cai(codons: list[str], w_table: dict[str, float]) -> float | None:
    """CAI for a codon list. Returns None if no contributing codons."""
    log_ws = []
    # Drop start codon (first codon)
    body = codons[1:] if len(codons) > 1 else []
    for c in body:
        if len(c) != 3 or c in STOP_CODONS:
            continue
        # Skip Met (ATG) and Trp (TGG)
        if c in {"ATG", "TGG"}:
            continue
        if c not in w_table:
            continue  # invalid (e.g. has N)
        wc = w_table[c]
        if wc <= 0:
            continue
        log_ws.append(math.log(wc))
    if not log_ws:
        return None
    return math.exp(sum(log_ws) / len(log_ws))


def parse_gff(path: Path):
    """Yield (seq_id, start_1based, end_inclusive, strand, partial, attrs_dict)."""
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            seq_id = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = dict(re.findall(r"(\w+)=([^;]+)", parts[8]))
            yield seq_id, start, end, strand, attrs.get("partial", "00"), attrs


def codons_of(seq: str) -> list[str]:
    return [seq[i:i + 3].upper() for i in range(0, len(seq) - 2, 3)]


def extract_sp_token(prompt: str) -> str:
    for tok in re.findall(r"<([^>]+)>", prompt):
        if tok.startswith("SP_"):
            return tok
    return "NONE"


def process_run(cache: Path, run: str, w_tables: dict[str, dict[str, float]],
                out_dir: Path) -> pd.DataFrame:
    print(f"\n=== {run} ===")
    gen = pd.read_parquet(cache / run / "generations.parquet").reset_index(drop=True)
    gff = cache / run / "orfs/prodigal.gff"
    if not gff.exists():
        raise SystemExit(f"missing gff: {gff}")
    # Per-row sequence lookup. Default seq_id = "gen_<idx>" (matches the gff
    # header convention used by eval/scripts/generate.py); preserve a pre-existing
    # seq_id column if present (e.g. for the training reference sample).
    if "seq_id" not in gen.columns:
        gen["seq_id"] = [f"gen_{i}" for i in range(len(gen))]
    gen["sp_token"] = gen["prompt"].apply(extract_sp_token)
    seq_by_id = dict(zip(gen["seq_id"], gen["sequence"]))

    rows = []
    skipped = defaultdict(int)
    for seq_id, start, end, strand, partial, attrs in parse_gff(gff):
        if partial != "00":
            skipped["partial"] += 1
            continue
        seq = seq_by_id.get(seq_id)
        if seq is None:
            skipped["seq_missing"] += 1
            continue
        nt = seq[start - 1:end]  # gff is 1-based inclusive
        if strand == "-":
            nt = revcomp(nt)
        if len(nt) % 3 != 0:
            skipped["frame_off"] += 1
            continue
        codons = codons_of(nt)
        # check internal stop codons in body (excluding the last codon)
        body_no_last = codons[1:-1]
        has_internal_stop = any(c in STOP_CODONS for c in body_no_last)
        if has_internal_stop:
            skipped["internal_stop"] += 1
            continue
        if len(codons) < MIN_CODONS + 1:  # +1 for start
            skipped["short"] += 1
            continue
        cai_values: dict[str, float | None] = {}
        for host, w in w_tables.items():
            cai_values[host] = cai(codons, w)
        rows.append({
            "seq_id": seq_id,
            "orf_id": attrs.get("ID", ""),
            "start": start,
            "end": end,
            "strand": strand,
            "length_aa": len(codons) - 1,  # minus stop
            "cai_ecoli": cai_values["ecoli"],
            "cai_yeast": cai_values["yeast"],
            "cai_human": cai_values["human"],
        })

    print(f"  ORFs kept: {len(rows)}  "
          f"(skipped: {dict(skipped)})")
    per_orf = pd.DataFrame(rows)
    per_orf.to_parquet(out_dir / f"cai_per_orf_{run}.parquet", index=False)

    # Per-sequence aggregation: mean (length-weighted), longest-ORF CAI, n_orfs
    per_seq_records = []
    for sid, group in per_orf.groupby("seq_id"):
        rec = {"seq_id": sid, "n_orfs": len(group)}
        weights = group["length_aa"].astype(float).values
        for host in HOSTS:
            col = f"cai_{host}"
            v = group[col].dropna().values
            if len(v) == 0:
                rec[f"{col}_mean"] = np.nan
                rec[f"{col}_wmean"] = np.nan
                rec[f"{col}_longest"] = np.nan
                continue
            valid_mask = group[col].notna().values
            ws = weights[valid_mask]
            vals = group[col].values[valid_mask]
            rec[f"{col}_mean"] = float(np.mean(vals))
            rec[f"{col}_wmean"] = float(np.average(vals, weights=ws)) if ws.sum() > 0 else float(np.nan)
            longest_idx = group["length_aa"].idxmax()
            rec[f"{col}_longest"] = float(group.loc[longest_idx, col]) if pd.notna(group.loc[longest_idx, col]) else np.nan
        per_seq_records.append(rec)
    per_seq = pd.DataFrame(per_seq_records)

    # Join SP token + prompt
    per_seq = per_seq.merge(gen[["seq_id", "prompt", "sp_token"]], on="seq_id", how="left")
    per_seq.to_parquet(out_dir / f"cai_per_seq_{run}.parquet", index=False)

    # Quick summary by SP token
    print("  per-SP CAI summary (length-weighted mean, n_seqs with ≥1 valid ORF):")
    for sp, sub in per_seq.dropna(subset=[f"cai_{h}_wmean" for h in HOSTS]).groupby("sp_token"):
        n = len(sub)
        means = {h: sub[f"cai_{h}_wmean"].mean() for h in HOSTS}
        print(f"    {sp:>16s}  n={n:>3d}   "
              f"ecoli={means['ecoli']:.3f}  yeast={means['yeast']:.3f}  human={means['human']:.3f}")
    return per_seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Names of runs (subdir of cache) to process")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Build w-tables
    w_tables: dict[str, dict[str, float]] = {}
    for nick, table_name in HOSTS.items():
        table = pct.get_codons_table(table_name)
        w_tables[nick] = build_w_table(table)
    print("loaded codon tables:")
    for h in HOSTS:
        print(f"  {h}: {HOSTS[h]}  ({sum(1 for v in w_tables[h].values() if v < 1.0)} sub-optimal codons)")

    for run in args.runs:
        process_run(args.cache, run, w_tables, args.out)


if __name__ == "__main__":
    main()
