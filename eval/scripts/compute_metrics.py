#!/usr/bin/env python3
"""Compute all eval metrics from generation + annotation artifacts.

Reads from a run directory containing:
  - generations.parquet (from generate.py)
  - plannotate_results.json (from annotate.py)
  - orfs/prodigal.gff (from annotate.py)
  - dustmasker/dustmasker.intervals (from annotate.py)

Outputs per-tier parquets and summary JSONs into metrics/ subdir.

Run in plannotate conda env:
    conda run -n plannotate python eval/scripts/compute_metrics.py \
        --run-dir eval/runs/firstlight_20260408 \
        --reference eval/reference
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Tier 1: Distributional metrics
# ---------------------------------------------------------------------------

def compute_kmer_counts(seq: str, k: int) -> dict[str, int]:
    """Count canonical k-mers (min of forward/revcomp)."""
    comp = str.maketrans("ATGC", "TACG")
    counts: dict[str, int] = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if "N" in kmer:
            continue
        rc = kmer.translate(comp)[::-1]
        canonical = min(kmer, rc)
        counts[canonical] += 1
    return dict(counts)


def parse_prodigal_gff(gff_path: str) -> dict[str, list[dict]]:
    """Parse Prodigal GFF output into per-sequence ORF lists."""
    orfs_by_seq: dict[str, list[dict]] = defaultdict(list)
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("//"):
                # Prodigal puts sequence boundaries as ## lines
                continue
            if "\t" not in line:
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            seq_id = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]

            # Check for valid start/stop
            has_start = "start_type=" in attrs
            partial = "partial=10" in attrs or "partial=01" in attrs or "partial=11" in attrs

            orfs_by_seq[seq_id].append({
                "start": start,
                "end": end,
                "strand": strand,
                "length_aa": (end - start + 1) // 3,
                "partial": partial,
            })
    return dict(orfs_by_seq)


def parse_dustmasker(intervals_path: str) -> dict[str, int]:
    """Parse dustmasker interval output into masked bp per sequence."""
    masked_bp: dict[str, int] = defaultdict(int)
    current_seq = None
    with open(intervals_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_seq = line[1:].strip()
            elif " - " in line and current_seq:
                parts = line.split(" - ")
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                masked_bp[current_seq] += end - start + 1
    return dict(masked_bp)


def compute_tier1(run_dir: Path, ref_dir: Path) -> pd.DataFrame:
    """Compute Tier 1 distributional metrics per generation."""
    from scipy.stats import ks_2samp, wasserstein_distance

    gen_df = pd.read_parquet(run_dir / "generations.parquet")

    # Load reference
    ref_metrics = pd.read_csv(ref_dir / "addgene_reference_metrics.csv")
    ref_seqs = pd.read_csv(ref_dir / "addgene_reference_500.csv")

    # Parse annotation artifacts
    gff_path = run_dir / "orfs" / "prodigal.gff"
    dust_path = run_dir / "dustmasker" / "dustmasker.intervals"

    orfs_by_seq = parse_prodigal_gff(str(gff_path)) if gff_path.exists() else {}
    masked_bp = parse_dustmasker(str(dust_path)) if dust_path.exists() else {}

    rows = []
    for idx, row in gen_df.iterrows():
        seq = row["sequence"]
        seq_id = f"gen_{idx}"
        length = len(seq)
        gc = row["gc"]

        # ORF stats
        orfs = orfs_by_seq.get(seq_id, [])
        n_orfs = len(orfs)
        orf_lengths = [o["length_aa"] for o in orfs]
        total_orf_bp = sum(o["length_aa"] * 3 for o in orfs)
        valid_orfs = [o for o in orfs if not o["partial"]]

        # k-mer counts
        kmer3 = compute_kmer_counts(seq, 3)
        kmer6 = compute_kmer_counts(seq, 6)

        # Low-complexity
        lc_bp = masked_bp.get(seq_id, 0)

        rows.append({
            "gen_id": idx,
            "length": length,
            "gc": gc,
            "n_orfs": n_orfs,
            "orf_density": total_orf_bp / max(length, 1),
            "pct_valid_orfs": len(valid_orfs) / max(n_orfs, 1),
            "mean_orf_length_aa": np.mean(orf_lengths) if orf_lengths else 0,
            "max_orf_length_aa": max(orf_lengths) if orf_lengths else 0,
            "lowcomp_frac": lc_bp / max(length, 1),
            "n_unique_3mers": len(kmer3),
            "n_unique_6mers": len(kmer6),
        })

    tier1_df = pd.DataFrame(rows)

    # Population-level stats vs reference
    summary = {}
    for metric in ["length", "gc"]:
        gen_vals = tier1_df[metric].values
        ref_vals = ref_metrics[metric].values if metric in ref_metrics.columns else ref_seqs["sequence"].str.len().values
        ks_stat, ks_p = ks_2samp(gen_vals, ref_vals)
        wd = wasserstein_distance(gen_vals, ref_vals)
        summary[metric] = {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "wasserstein": float(wd)}

    # Save
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tier1_df.to_parquet(metrics_dir / "tier1_distributional.parquet", index=False)
    with open(metrics_dir / "tier1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Tier 1: {len(tier1_df)} rows -> {metrics_dir / 'tier1_distributional.parquet'}")
    return tier1_df


# ---------------------------------------------------------------------------
# Tier 3: Essentials (pLannotate-based)
# ---------------------------------------------------------------------------

def compute_tier3(run_dir: Path, config_dir: Path) -> pd.DataFrame:
    """Compute Tier 3 essential feature booleans from pLannotate output."""
    plannotate_path = run_dir / "plannotate_results.json"
    if not plannotate_path.exists():
        print("Tier 3: No plannotate_results.json found, skipping")
        return pd.DataFrame()

    with open(plannotate_path) as f:
        plannotate_results = json.load(f)

    # Load feature categories
    cats_path = config_dir / "feature_categories.yaml"
    with open(cats_path) as f:
        categories = yaml.safe_load(f)

    def classify_feature(feat: dict) -> set[str]:
        """Return which categories a single feature belongs to."""
        matched = set()
        ftype = feat.get("Type", "").lower()
        fname = feat.get("Feature", "").lower()
        fsseq = feat.get("sseqid", "").lower()

        for cat_name, cat_def in categories.items():
            # Match by plannotate type
            for pt in cat_def.get("plannotate_types", []):
                if pt.lower() == ftype:
                    matched.add(cat_name)
            # Match by keyword in feature name or sseqid
            for kw in cat_def.get("keywords", []):
                if kw.lower() in fname or kw.lower() in fsseq:
                    matched.add(cat_name)
        return matched

    rows = []
    for result in plannotate_results:
        features = result.get("features", [])
        seq_len = result.get("seq_length", 0)
        coverage_bp = result.get("coverage_bp", 0)

        all_cats = set()
        for feat in features:
            all_cats |= classify_feature(feat)

        has_ori = "origin" in all_cats
        has_marker = "selection_marker" in all_cats
        has_prom = "promoter" in all_cats
        has_term = "terminator" in all_cats
        has_cds = "cds" in all_cats

        # Plausibility: has at least origin + marker + one expression element
        plausible = has_ori and has_marker and (has_prom or has_cds)

        rows.append({
            "seq_id": result["seq_id"],
            "n_features": result["n_features"],
            "coverage_frac": coverage_bp / max(seq_len, 1),
            "has_ori": has_ori,
            "has_selection_marker": has_marker,
            "has_promoter": has_prom,
            "has_terminator": has_term,
            "has_cds": has_cds,
            "plausibility_pass": plausible,
        })

    tier3_df = pd.DataFrame(rows)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tier3_df.to_parquet(metrics_dir / "tier3_essentials.parquet", index=False)

    print(f"Tier 3: {len(tier3_df)} rows, "
          f"plausibility rate = {tier3_df['plausibility_pass'].mean():.1%}")
    return tier3_df


# ---------------------------------------------------------------------------
# Tier 5: Architecture (promoter-CDS-terminator context)
# ---------------------------------------------------------------------------

def compute_tier5(run_dir: Path, config_dir: Path) -> pd.DataFrame:
    """Compute Tier 5 architecture metrics from pLannotate output."""
    plannotate_path = run_dir / "plannotate_results.json"
    if not plannotate_path.exists():
        print("Tier 5: No plannotate_results.json found, skipping")
        return pd.DataFrame()

    with open(plannotate_path) as f:
        plannotate_results = json.load(f)

    with open(config_dir / "feature_categories.yaml") as f:
        categories = yaml.safe_load(f)

    def get_category(feat: dict) -> str | None:
        ftype = feat.get("Type", "").lower()
        fname = feat.get("Feature", "").lower()
        fsseq = feat.get("sseqid", "").lower()
        for cat_name, cat_def in categories.items():
            for pt in cat_def.get("plannotate_types", []):
                if pt.lower() == ftype:
                    return cat_name
            for kw in cat_def.get("keywords", []):
                if kw.lower() in fname or kw.lower() in fsseq:
                    return cat_name
        return None

    rows = []
    for result in plannotate_results:
        features = result.get("features", [])

        # Categorize features with positions
        typed_features = []
        for feat in features:
            cat = get_category(feat)
            if cat:
                typed_features.append({
                    "category": cat,
                    "start": feat.get("qstart", 0),
                    "end": feat.get("qend", 0),
                })

        # Count CDSs with valid promoter/terminator context (within 500bp, same region)
        cds_feats = [f for f in typed_features if f["category"] == "cds"]
        prom_feats = [f for f in typed_features if f["category"] == "promoter"]
        term_feats = [f for f in typed_features if f["category"] == "terminator"]

        cds_with_promoter = 0
        cds_with_terminator = 0
        context_window = 500

        for cds in cds_feats:
            # Check for promoter upstream (within 500bp before CDS start)
            for prom in prom_feats:
                if abs(cds["start"] - prom["end"]) < context_window:
                    cds_with_promoter += 1
                    break
            # Check for terminator downstream (within 500bp after CDS end)
            for term in term_feats:
                if abs(term["start"] - cds["end"]) < context_window:
                    cds_with_terminator += 1
                    break

        # Check for overlapping features
        n_overlapping = 0
        for i, f1 in enumerate(typed_features):
            for f2 in typed_features[i + 1:]:
                if f1["start"] < f2["end"] and f2["start"] < f1["end"]:
                    n_overlapping += 1

        n_origins = sum(1 for f in typed_features if f["category"] == "origin")
        n_markers = sum(1 for f in typed_features if f["category"] == "selection_marker")

        rows.append({
            "seq_id": result["seq_id"],
            "n_cds": len(cds_feats),
            "n_cds_with_promoter": cds_with_promoter,
            "n_cds_with_terminator": cds_with_terminator,
            "pct_cds_with_promoter": cds_with_promoter / max(len(cds_feats), 1),
            "pct_cds_with_terminator": cds_with_terminator / max(len(cds_feats), 1),
            "n_overlapping_features": n_overlapping,
            "n_origins": n_origins,
            "n_selection_markers": n_markers,
        })

    tier5_df = pd.DataFrame(rows)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tier5_df.to_parquet(metrics_dir / "tier5_architecture.parquet", index=False)

    print(f"Tier 5: {len(tier5_df)} rows")
    return tier5_df


# ---------------------------------------------------------------------------
# Memorization (sourmash + optional minimap2 follow-up)
# ---------------------------------------------------------------------------

def compute_memorization(run_dir: Path, ref_dir: Path) -> pd.DataFrame:
    """Check for memorization using sourmash containment against training set."""
    fasta_path = run_dir / "generations.fasta"
    training_sig = ref_dir / "training_sigs.zip"

    if not fasta_path.exists() or not training_sig.exists():
        print("Memorization: Missing fasta or training sig, skipping")
        return pd.DataFrame()

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Sketch generations
    gen_sig = metrics_dir / "gen_sigs.zip"
    subprocess.run([
        "sourmash", "sketch", "dna",
        "-p", "k=31,scaled=100",
        str(fasta_path), "-o", str(gen_sig),
        "--singleton",  # one sig per sequence
    ], capture_output=True, text=True)

    # Search against training
    search_out = metrics_dir / "memorization_search.csv"
    result = subprocess.run([
        "sourmash", "search",
        str(gen_sig), str(training_sig),
        "--containment", "--threshold", "0.3",
        "-o", str(search_out),
    ], capture_output=True, text=True)

    if search_out.exists():
        mem_df = pd.read_csv(search_out)
        n_high = (mem_df["similarity"] > 0.8).sum() if "similarity" in mem_df.columns else 0
        print(f"Memorization: {len(mem_df)} hits above 0.3 containment, "
              f"{n_high} above 0.8 (flagged)")
    else:
        mem_df = pd.DataFrame()
        print("Memorization: no hits found")

    mem_df.to_parquet(metrics_dir / "memorization.parquet", index=False)
    return mem_df


# ---------------------------------------------------------------------------
# Discriminator (LightGBM real vs generated)
# ---------------------------------------------------------------------------

def compute_discriminator(run_dir: Path, ref_dir: Path) -> dict:
    """Train a LightGBM classifier to distinguish real vs generated sequences."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    metrics_dir = run_dir / "metrics"
    tier1_path = metrics_dir / "tier1_distributional.parquet"
    tier3_path = metrics_dir / "tier3_essentials.parquet"

    if not tier1_path.exists():
        print("Discriminator: No tier1 metrics, skipping")
        return {}

    gen_t1 = pd.read_parquet(tier1_path)
    gen_t1["label"] = 1  # generated

    # Build features for real sequences from reference
    ref_metrics = pd.read_csv(ref_dir / "addgene_reference_metrics.csv")
    ref_seqs = pd.read_csv(ref_dir / "addgene_reference_500.csv")

    # Compute matching features for real sequences
    real_rows = []
    for _, row in ref_seqs.iterrows():
        seq = row["sequence"]
        kmer3 = compute_kmer_counts(seq, 3)
        kmer6 = compute_kmer_counts(seq, 6)
        real_rows.append({
            "length": len(seq),
            "gc": (seq.count("G") + seq.count("C")) / max(len(seq), 1),
            "n_unique_3mers": len(kmer3),
            "n_unique_6mers": len(kmer6),
            "lowcomp_frac": 0,  # would need dustmasker on ref
            "orf_density": 0,   # would need prodigal on ref
            "label": 0,
        })

    real_df = pd.DataFrame(real_rows)

    # Use overlapping features
    feature_cols = ["length", "gc", "n_unique_3mers", "n_unique_6mers", "lowcomp_frac", "orf_density"]
    available_cols = [c for c in feature_cols if c in gen_t1.columns and c in real_df.columns]

    combined = pd.concat([
        gen_t1[available_cols + ["label"]],
        real_df[available_cols + ["label"]],
    ], ignore_index=True)

    X = combined[available_cols].values
    y = combined["label"].values

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=available_cols)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    model = lgb.train(params, dtrain, num_boost_round=200)

    # Evaluate
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    # Feature importance
    importance = dict(zip(available_cols, model.feature_importance(importance_type="gain").tolist()))

    result = {
        "auc": float(auc),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "feature_importance": importance,
    }

    with open(metrics_dir / "discriminator.json", "w") as f:
        json.dump(result, f, indent=2)

    # SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X_test, feature_names=available_cols, show=False)
        plt.tight_layout()
        plt.savefig(metrics_dir / "shap_summary.png", dpi=150)
        plt.close()
        print(f"Discriminator: SHAP plot saved")
    except Exception as e:
        print(f"Discriminator: SHAP plot failed: {e}")

    print(f"Discriminator: AUC = {auc:.3f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference", default="eval/reference")
    parser.add_argument("--config", default="eval/config")
    parser.add_argument("--tiers", default="1,3,5,mem,disc",
                        help="Comma-separated list of metrics to compute")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ref_dir = Path(args.reference)
    config_dir = Path(args.config)
    tiers = set(args.tiers.split(","))

    print(f"Computing metrics for {run_dir}")
    print(f"  Reference: {ref_dir}")
    print(f"  Tiers: {tiers}")

    if "1" in tiers:
        compute_tier1(run_dir, ref_dir)
    if "3" in tiers:
        compute_tier3(run_dir, config_dir)
    if "5" in tiers:
        compute_tier5(run_dir, config_dir)
    if "mem" in tiers:
        compute_memorization(run_dir, ref_dir)
    if "disc" in tiers:
        compute_discriminator(run_dir, ref_dir)

    print(f"\nAll metrics complete. Results in {run_dir / 'metrics'}")


if __name__ == "__main__":
    main()
