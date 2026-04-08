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
# ---------------------------------------------------------------------------

# Map prompt tokens to pLannotate feature keywords
TOKEN_TO_PLANNOTATE = {
    # Origins
    "ORI_COLE1": ["ColE1", "pBR322", "pUC ori"],
    "ORI_F1": ["f1 ori", "F1 ori"],
    "ORI_SV40": ["SV40 ori"],
    "ORI_RSF": ["RSF"],
    "ORI_2MU": ["2 micron", "2mu"],
    "ORI_P15A": ["p15A"],
    "ORI_PBLUESCRIPT": ["pBluescript"],
    "ORI_RK2": ["RK2", "oriV"],
    # AMR
    "AMR_AMPICILLIN": ["AmpR", "bla", "ampicillin", "beta-lactamase"],
    "AMR_KANAMYCIN": ["KanR", "aph", "kanamycin", "NeoR", "neomycin"],
    "AMR_CHLORAMPHENICOL": ["CmR", "cat", "chloramphenicol"],
    "AMR_SPECTINOMYCIN": ["SpecR", "aadA", "spectinomycin"],
    "AMR_GENTAMICIN": ["GentR", "aacC", "gentamicin"],
    "AMR_TETRACYCLINE": ["TetR", "tetracycline"],
    "AMR_HYGROMYCIN": ["HygR", "hygromycin"],
    "AMR_PUROMYCIN": ["PuroR", "puromycin"],
    "AMR_ZEOCIN": ["ZeoR", "ble", "zeocin"],
    "AMR_NEOMYCIN": ["NeoR", "neo", "neomycin", "G418"],
    "AMR_BLASTICIDIN": ["BlastR", "blasticidin", "bsd"],
    "AMR_NOURSEOTHRICIN": ["NatR", "nourseothricin"],
    # Promoters
    "PROM_CMV": ["CMV promoter", "CMV enhancer", "hCMV"],
    "PROM_SV40": ["SV40 promoter", "SV40 early"],
    "PROM_AMPR": ["AmpR promoter", "bla promoter"],
    "PROM_T7": ["T7 promoter"],
    "PROM_T3": ["T3 promoter"],
    "PROM_SP6": ["SP6 promoter"],
    "PROM_LAC": ["lac promoter", "lac operator", "lacO"],
    "PROM_EF1A": ["EF-1", "EF1", "EFS"],
    "PROM_CAG": ["CAG", "CAGGS"],
    "PROM_PGK": ["PGK promoter"],
    "PROM_U6": ["U6 promoter"],
    "PROM_UBC": ["UBC", "UbC"],
    # Elements
    "ELEM_IRES": ["IRES"],
    "ELEM_POLYA_SV40": ["SV40 poly", "SV40 pA", "SV40 late"],
    "ELEM_POLYA_BGH": ["BGH poly", "BGH pA"],
    "ELEM_CMV_ENHANCER": ["CMV enhancer"],
    "ELEM_CMV_INTRON": ["CMV intron", "chimeric intron"],
    "ELEM_WPRE": ["WPRE"],
    "ELEM_KOZAK": ["Kozak"],
    "ELEM_TRACRRNA": ["tracrRNA", "gRNA scaffold"],
    # Reporters
    "REPORTER_EGFP": ["EGFP", "eGFP"],
    "REPORTER_GFP": ["GFP"],
    "REPORTER_MCHERRY": ["mCherry"],
    "REPORTER_MEMERALD": ["mEmerald"],
    "REPORTER_YFP": ["YFP", "EYFP"],
    "REPORTER_BFP": ["BFP", "EBFP"],
    "REPORTER_LUCIFERASE": ["luciferase", "Luc2"],
    # Tags
    "TAG_FLAG": ["FLAG"],
    "TAG_HA": ["HA tag"],
    "TAG_V5": ["V5 tag"],
    "TAG_HIS": ["6xHis", "His tag", "polyhistidine"],
    "TAG_GST": ["GST"],
    "TAG_MYC": ["Myc", "c-Myc tag"],
    "TAG_STREP": ["Strep", "StrepII"],
    "TAG_MBP": ["MBP"],
}

# Categories we can verify (excludes metadata tokens like VEC_, SP_, COPY_, SIZE_, GC_)
VERIFIABLE_PREFIXES = {"ORI", "AMR", "PROM", "ELEM", "REPORTER", "TAG"}


def compute_prompt_fidelity(run_dir: Path) -> dict:
    """Check how often requested features appear in pLannotate output."""
    gen_df = pd.read_parquet(run_dir / "generations.parquet")
    pn_path = run_dir / "plannotate_results.json"
    if not pn_path.exists():
        print("Prompt fidelity: no plannotate results, skipping")
        return {}

    with open(pn_path) as f:
        pn_results = json.load(f)

    # Build per-sequence feature text for searching
    seq_features = {}
    for result in pn_results:
        sid = result["seq_id"]
        feat_text = " ".join(
            f"{f.get('Feature', '')} {f.get('sseqid', '')} {f.get('Type', '')}"
            for f in result.get("features", [])
        ).lower()
        seq_features[sid] = feat_text

    # For each sequence, check which requested tokens are found
    per_token_hits = defaultdict(lambda: {"requested": 0, "found": 0})
    per_category_hits = defaultdict(lambda: {"requested": 0, "found": 0})
    per_seq_fidelity = []

    for idx, row in gen_df.iterrows():
        prompt_tokens = re.findall(r"<([^>]+)>", row["prompt"])
        seq_id = f"gen_{idx}"
        feat_text = seq_features.get(seq_id, "")

        n_requested = 0
        n_found = 0

        for tok in prompt_tokens:
            prefix = tok.split("_", 1)[0]
            if prefix not in VERIFIABLE_PREFIXES:
                continue

            n_requested += 1
            per_token_hits[tok]["requested"] += 1
            category = prefix
            per_category_hits[category]["requested"] += 1

            # Check if any keyword matches
            keywords = TOKEN_TO_PLANNOTATE.get(tok, [])
            found = any(kw.lower() in feat_text for kw in keywords)

            if found:
                n_found += 1
                per_token_hits[tok]["found"] += 1
                per_category_hits[category]["found"] += 1

        if n_requested > 0:
            per_seq_fidelity.append(n_found / n_requested)

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
        "overall_fidelity": float(overall_fidelity),
        "n_sequences": len(per_seq_fidelity),
        "per_category": category_rates,
        "per_token": token_rates,
    }

    print(f"Prompt fidelity: {overall_fidelity:.1%} overall")
    print(f"  Per category:")
    for cat, r in sorted(category_rates.items()):
        print(f"    {cat}: {r['hit_rate']:.1%} ({r['n_found']}/{r['n_requested']})")

    return result


# ---------------------------------------------------------------------------
# ViennaRNA MFE density
# ---------------------------------------------------------------------------

def compute_mfe_density(run_dir: Path) -> dict:
    """Compute MFE density for generated sequences using ViennaRNA."""
    try:
        import RNA
    except ImportError:
        print("MFE: ViennaRNA not installed, skipping")
        return {}

    gen_df = pd.read_parquet(run_dir / "generations.parquet")

    mfe_densities = []
    for idx, row in gen_df.iterrows():
        seq = row["sequence"]
        if len(seq) < 50 or len(seq) > 15000:
            mfe_densities.append(None)
            continue
        # For long sequences, compute MFE on a sliding window and average
        if len(seq) > 5000:
            # Sample 5 windows of 2000bp
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

        if (idx + 1) % 100 == 0:
            print(f"  MFE: {idx+1}/{len(gen_df)}")

    valid = [m for m in mfe_densities if m is not None]
    result = {
        "mean_mfe_density": float(np.mean(valid)),
        "median_mfe_density": float(np.median(valid)),
        "std_mfe_density": float(np.std(valid)),
        "n_computed": len(valid),
    }
    print(f"MFE density: mean={np.mean(valid):.4f} kcal/mol/nt (n={len(valid)})")
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
        print("Codon usage: no prodigal output, skipping")
        return {}

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
        print("Codon usage: no ORFs found")
        return {}

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

    # Save
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "extra_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {metrics_dir / 'extra_metrics.json'}")


if __name__ == "__main__":
    main()
