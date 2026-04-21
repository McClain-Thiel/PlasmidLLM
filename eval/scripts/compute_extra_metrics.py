#!/usr/bin/env python3
"""Compute additional metrics: prompt fidelity, MFE density, codon usage, GC skew.

Must run in plannotate conda env (needs ViennaRNA, BioPython).

Usage:
    conda run -n plannotate python eval/scripts/compute_extra_metrics.py \
        --run-dir eval/runs/grpo_plannotate_full_20260408
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prompt fidelity: does the generated sequence contain requested features?
#
# A "hit" is sseqid set intersection against the motif registry: for each
# conditioning token (e.g. <AMR_KANAMYCIN>) the registry lists every pLannotate
# sseqid that counts as a valid detection. A generated sequence is a hit on
# that token iff pLannotate emits at least one feature whose sseqid is in the
# registry set for that token. No substring / keyword matching.
#
# Tokens that are not in the registry (e.g. <COPY_HIGH>, <VEC_MAMMALIAN>,
# <SIZE_LARGE>) are metadata, not pLannotate-verifiable, and are ignored.
# ---------------------------------------------------------------------------

DEFAULT_MOTIF_REGISTRY = (
    Path(__file__).resolve().parents[2] / "data" / "motif_registry_combined.parquet"
)


def load_token_sseqid_map(registry_path: Path) -> dict[str, set[str]]:
    """Return {token_name_without_brackets: set[sseqid]} from the motif registry."""
    df = pd.read_parquet(registry_path, columns=["token", "sseqid"])
    if df.empty:
        raise ValueError(f"motif registry is empty: {registry_path}")
    mapping: dict[str, set[str]] = defaultdict(set)
    for tok, sid in zip(df["token"], df["sseqid"]):
        if not isinstance(tok, str) or not isinstance(sid, str):
            continue
        mapping[tok.strip("<>")].add(sid)
    return dict(mapping)


def compute_prompt_fidelity(run_dir: Path, motif_registry: Path | None = None) -> dict:
    """sseqid-based prompt fidelity via the motif registry.

    Loads `motif_registry_combined.parquet` (or the path given), builds the
    per-token sseqid set, and scores each generation by set intersection with
    the sseqids pLannotate emitted for that sequence.
    """
    gen_df = pd.read_parquet(run_dir / "generations.parquet")
    pn_path = run_dir / "plannotate_results.json"
    if not pn_path.exists():
        raise FileNotFoundError(
            f"plannotate_results.json missing at {pn_path}; run annotate.py first"
        )

    with open(pn_path) as f:
        pn_results = json.load(f)

    registry_path = Path(motif_registry) if motif_registry else DEFAULT_MOTIF_REGISTRY
    token_sseqids = load_token_sseqid_map(registry_path)
    verifiable_tokens = set(token_sseqids.keys())
    verifiable_categories = {t.split("_", 1)[0] for t in verifiable_tokens}

    # Per-sequence set of pLannotate sseqids.
    gen_sseqids: dict[str, set[str]] = {}
    for result in pn_results:
        sid = result["seq_id"]
        gen_sseqids[sid] = {
            f["sseqid"] for f in result.get("features", [])
            if isinstance(f.get("sseqid"), str) and f["sseqid"]
        }

    # Aggregate counters + per-sequence rows (one row per prompt so downstream
    # paired analyses can McNemar / bootstrap against the matched base run).
    per_token_hits = defaultdict(lambda: {"requested": 0, "found": 0})
    per_category_hits = defaultdict(lambda: {"requested": 0, "found": 0})
    per_seq_fidelity: list[float] = []
    per_seq_rows: list[dict] = []

    for idx, row in gen_df.iterrows():
        prompt_tokens = re.findall(r"<([^>]+)>", row["prompt"])
        seq_id = f"gen_{idx}"
        hit_sseqids = gen_sseqids.get(seq_id, set())

        requested_toks: list[str] = []
        found_toks: list[str] = []
        cat_req: dict[str, int] = defaultdict(int)
        cat_fnd: dict[str, int] = defaultdict(int)

        for tok in prompt_tokens:
            if tok not in verifiable_tokens:
                continue
            category = tok.split("_", 1)[0]
            per_token_hits[tok]["requested"] += 1
            per_category_hits[category]["requested"] += 1
            requested_toks.append(tok)
            cat_req[category] += 1

            if token_sseqids[tok] & hit_sseqids:
                per_token_hits[tok]["found"] += 1
                per_category_hits[category]["found"] += 1
                found_toks.append(tok)
                cat_fnd[category] += 1

        n_requested = len(requested_toks)
        n_found = len(found_toks)
        if n_requested > 0:
            per_seq_fidelity.append(n_found / n_requested)

        row_record = {
            "seq_id": seq_id,
            "prompt_index": int(idx),
            "prompt": row["prompt"],
            "n_requested": n_requested,
            "n_found": n_found,
            "fidelity": (n_found / n_requested) if n_requested > 0 else float("nan"),
            "perfect_match": bool(n_requested > 0 and n_found == n_requested),
            "requested_tokens": requested_toks,
            "found_tokens": found_toks,
        }
        for cat in verifiable_categories:
            row_record[f"n_requested_{cat}"] = int(cat_req.get(cat, 0))
            row_record[f"n_found_{cat}"] = int(cat_fnd.get(cat, 0))
        per_seq_rows.append(row_record)

    # Aggregate
    overall_fidelity = np.mean(per_seq_fidelity) if per_seq_fidelity else 0.0

    token_rates = {}
    for tok, counts in sorted(per_token_hits.items()):
        if counts["requested"] >= 5:  # only report tokens requested at least 5 times
            token_rates[tok] = {
                "hit_rate": counts["found"] / counts["requested"],
                "n_requested": counts["requested"],
                "n_found": counts["found"],
            }

    category_rates = {}
    for cat, counts in sorted(per_category_hits.items()):
        category_rates[cat] = {
            "hit_rate": counts["found"] / max(counts["requested"], 1),
            "n_requested": counts["requested"],
            "n_found": counts["found"],
        }

    result = {
        "method": "sseqid_intersection_via_motif_registry",
        "motif_registry": str(registry_path),
        "n_verifiable_tokens": len(verifiable_tokens),
        "overall_fidelity": float(overall_fidelity),
        "n_sequences": len(per_seq_fidelity),
        "per_category": category_rates,
        "per_token": token_rates,
    }

    per_seq_df = pd.DataFrame(per_seq_rows)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_seq_df.to_parquet(metrics_dir / "prompt_fidelity_per_seq.parquet", index=False)

    print(f"Prompt fidelity: {overall_fidelity:.1%} overall "
          f"(per-seq rows: {len(per_seq_df)}, rows with any hard tokens: "
          f"{int((per_seq_df['n_requested'] > 0).sum())})")
    print(f"  Per category:")
    for cat, r in sorted(category_rates.items()):
        print(f"    {cat}: {r['hit_rate']:.1%} ({r['n_found']}/{r['n_requested']})")

    return result


# ---------------------------------------------------------------------------
# ViennaRNA MFE density
# ---------------------------------------------------------------------------

def compute_mfe_density(run_dir: Path) -> dict:
    """Compute DNA MFE density using ViennaRNA with Mathews2004 DNA parameters.

    ViennaRNA (the `RNA` Python module) is a hard dependency of this metric. If
    it is not installed the caller must skip MFE via ``--metrics`` rather than
    relying on a silent empty-dict return from this function.
    """
    import RNA

    # Load DNA energy parameters (not RNA — these are DNA plasmids!)
    RNA.params_load_DNA_Mathews2004()
    print("  Using DNA parameters (Mathews et al. 2004)")

    gen_df = pd.read_parquet(run_dir / "generations.parquet")

    mfe_densities = []
    for idx, row in gen_df.iterrows():
        seq = row["sequence"]
        if len(seq) < 50 or len(seq) > 15000:
            mfe_densities.append(None)
            continue
        try:
            # For long sequences, compute MFE on sliding windows and average
            if len(seq) > 5000:
                rng = np.random.RandomState(idx)
                window_mfes = []
                for _ in range(5):
                    start = rng.randint(0, len(seq) - 2000)
                    window = seq[start:start + 2000]
                    _, mfe = RNA.fold(window)
                    window_mfes.append(mfe / len(window))
                mfe_densities.append(float(np.mean(window_mfes)))
            else:
                _, mfe = RNA.fold(seq)
                mfe_densities.append(mfe / len(seq))
        except Exception as e:
            mfe_densities.append(None)
            if idx < 5:
                print(f"  MFE error at idx {idx}: {e}")

        if (idx + 1) % 100 == 0:
            print(f"  MFE: {idx+1}/{len(gen_df)}")

    valid = [m for m in mfe_densities if m is not None]
    result = {
        "mean_mfe_density": float(np.mean(valid)),
        "median_mfe_density": float(np.median(valid)),
        "std_mfe_density": float(np.std(valid)),
        "n_computed": len(valid),
        "parameter_set": "DNA_Mathews2004",
    }
    print(f"MFE density (DNA): mean={np.mean(valid):.4f} kcal/mol/nt (n={len(valid)})")
    return result


# ---------------------------------------------------------------------------
# Codon usage quality (for Prodigal-predicted ORFs)
# ---------------------------------------------------------------------------

def compute_codon_usage(run_dir: Path) -> dict:
    """Analyze codon usage in predicted ORFs."""
    from Bio import SeqIO
    from Bio.Seq import Seq

    protein_path = run_dir / "orfs" / "prodigal_proteins.faa"
    gff_path = run_dir / "orfs" / "prodigal.gff"

    if not gff_path.exists():
        raise FileNotFoundError(
            f"codon usage metric requires prodigal output at {gff_path}; "
            "run eval/scripts/annotate.py first"
        )

    # Parse ORF sequences from the FASTA and GFF
    gen_fasta = run_dir / "generations.fasta"
    seqs = {r.id: str(r.seq) for r in SeqIO.parse(str(gen_fasta), "fasta")}

    # Standard codon table
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    # E. coli codon frequencies (common host for plasmids)
    ECOLI_FREQ = {
        'TTT': 0.58, 'TTC': 0.42, 'TTA': 0.14, 'TTG': 0.13,
        'CTT': 0.12, 'CTC': 0.10, 'CTA': 0.04, 'CTG': 0.47,
        'ATT': 0.51, 'ATC': 0.42, 'ATA': 0.08, 'ATG': 1.00,
        'GTT': 0.28, 'GTC': 0.20, 'GTA': 0.17, 'GTG': 0.35,
        'TCT': 0.17, 'TCC': 0.15, 'TCA': 0.14, 'TCG': 0.14,
        'CCT': 0.18, 'CCC': 0.13, 'CCA': 0.20, 'CCG': 0.49,
        'ACT': 0.19, 'ACC': 0.40, 'ACA': 0.17, 'ACG': 0.25,
        'GCT': 0.18, 'GCC': 0.26, 'GCA': 0.23, 'GCG': 0.33,
        'TAT': 0.59, 'TAC': 0.41,
        'CAT': 0.57, 'CAC': 0.43, 'CAA': 0.34, 'CAG': 0.66,
        'AAT': 0.49, 'AAC': 0.51, 'AAA': 0.74, 'AAG': 0.26,
        'GAT': 0.63, 'GAC': 0.37, 'GAA': 0.68, 'GAG': 0.32,
        'TGT': 0.46, 'TGC': 0.54, 'TGG': 1.00,
        'CGT': 0.36, 'CGC': 0.36, 'CGA': 0.07, 'CGG': 0.11,
        'AGT': 0.16, 'AGC': 0.25, 'AGA': 0.07, 'AGG': 0.04,
        'GGT': 0.35, 'GGC': 0.37, 'GGA': 0.13, 'GGG': 0.15,
    }

    # Extract ORF DNA sequences from GFF
    codon_counts = Counter()
    n_orfs = 0
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            seq_id = parts[0]
            start = int(parts[3]) - 1  # GFF is 1-based
            end = int(parts[4])
            strand = parts[6]

            if seq_id not in seqs:
                continue
            orf_seq = seqs[seq_id][start:end]
            if strand == "-":
                orf_seq = str(Seq(orf_seq).reverse_complement())

            # Count codons
            for i in range(0, len(orf_seq) - 2, 3):
                codon = orf_seq[i:i+3].upper()
                if codon in CODON_TABLE:
                    codon_counts[codon] += 1
            n_orfs += 1

    if n_orfs == 0:
        raise RuntimeError(
            f"codon usage: no ORFs parsed from {gff_path}; "
            "prodigal output is empty, which is unexpected for plasmid generations"
        )

    total_codons = sum(codon_counts.values())

    # Compute codon adaptation index (simplified)
    # CAI = geometric mean of relative adaptiveness (w) for each codon
    # w = freq(codon) / max(freq(synonymous codons))
    gen_freq = {c: codon_counts.get(c, 0) / max(total_codons, 1) for c in CODON_TABLE}

    # Group by amino acid
    aa_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        if aa != "*":
            aa_codons[aa].append(codon)

    # Jensen-Shannon divergence between generated and E. coli codon usage
    from scipy.stats import entropy
    from scipy.spatial.distance import jensenshannon

    # Per-amino-acid JSD
    aa_jsds = []
    for aa, codons in aa_codons.items():
        if len(codons) <= 1:
            continue
        gen_probs = np.array([gen_freq.get(c, 1e-10) for c in codons])
        gen_probs = gen_probs / gen_probs.sum()
        eco_probs = np.array([ECOLI_FREQ.get(c, 1e-10) for c in codons])
        eco_probs = eco_probs / eco_probs.sum()
        aa_jsds.append(jensenshannon(gen_probs, eco_probs))

    result = {
        "n_orfs_analyzed": n_orfs,
        "total_codons": total_codons,
        "mean_codon_jsd_vs_ecoli": float(np.mean(aa_jsds)),
        "median_codon_jsd_vs_ecoli": float(np.median(aa_jsds)),
        "top_codons": {c: round(v, 4) for c, v in codon_counts.most_common(10)},
    }
    print(f"Codon usage: {n_orfs} ORFs, {total_codons} codons, "
          f"JSD vs E.coli = {np.mean(aa_jsds):.4f}")
    return result


# ---------------------------------------------------------------------------
# GC skew
# ---------------------------------------------------------------------------

def compute_gc_skew(run_dir: Path) -> dict:
    """Compute GC skew statistics for generated sequences."""
    gen_df = pd.read_parquet(run_dir / "generations.parquet")

    skew_magnitudes = []
    for _, row in gen_df.iterrows():
        seq = row["sequence"].upper()
        if len(seq) < 1000:
            continue
        # Compute GC skew in windows
        window = 1000
        step = 500
        skews = []
        for i in range(0, len(seq) - window, step):
            w = seq[i:i + window]
            g = w.count("G")
            c = w.count("C")
            if g + c > 0:
                skews.append((g - c) / (g + c))
        if skews:
            skew_magnitudes.append(np.std(skews))  # variation in skew = structure

    result = {
        "mean_skew_variation": float(np.mean(skew_magnitudes)),
        "median_skew_variation": float(np.median(skew_magnitudes)),
        "n_computed": len(skew_magnitudes),
    }
    print(f"GC skew: mean variation={np.mean(skew_magnitudes):.4f} (n={len(skew_magnitudes)})")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics", default="fidelity,mfe,codon,skew",
                        help="Comma-separated metrics to compute")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = set(args.metrics.split(","))
    results = {}

    if "fidelity" in metrics:
        print("=== Prompt Fidelity ===")
        results["prompt_fidelity"] = compute_prompt_fidelity(run_dir)
        print()

    if "mfe" in metrics:
        print("=== MFE Density ===")
        results["mfe_density"] = compute_mfe_density(run_dir)
        print()

    if "codon" in metrics:
        print("=== Codon Usage ===")
        results["codon_usage"] = compute_codon_usage(run_dir)
        print()

    if "skew" in metrics:
        print("=== GC Skew ===")
        results["gc_skew"] = compute_gc_skew(run_dir)
        print()

    # Save (merge with existing results, don't overwrite)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "extra_metrics.json"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved to {out_path} (keys: {list(existing.keys())})")


if __name__ == "__main__":
    main()
