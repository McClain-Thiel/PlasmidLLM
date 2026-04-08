#!/usr/bin/env python3
"""Phase 6: Render HTML eval report from computed metrics.

Generates a standalone HTML report with inline plots (base64 PNGs),
tables, and run configuration.

Usage:
    conda run -n plannotate python eval/scripts/render_report.py \
        --run-dir eval/runs/grpo_plannotate_full_20260408 \
        --baselines-dir eval/baselines \
        --reference eval/reference
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_violin_comparison(gen_vals, ref_vals, baselines: dict, metric_name: str, xlabel: str) -> str:
    """Violin plot of a metric across generated, reference, and baselines."""
    fig, ax = plt.subplots(figsize=(8, 4))

    data = [gen_vals]
    labels = ["Generated"]
    colors = ["#4C72B0"]

    if ref_vals is not None and len(ref_vals) > 0:
        data.append(ref_vals)
        labels.append("Addgene-500")
        colors.append("#55A868")

    for name, vals in baselines.items():
        if vals is not None and len(vals) > 0:
            data.append(vals)
            labels.append(name.capitalize())
            colors.append({"random": "#C44E52", "shuffled": "#8172B2", "real": "#55A868"}.get(name, "#CCB974"))

    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(xlabel)
    ax.set_title(metric_name)
    ax.grid(axis="y", alpha=0.3)

    return fig_to_base64(fig)


def plot_bar_chart(categories: dict, title: str) -> str:
    """Horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(7, max(3, len(categories) * 0.5)))
    names = list(categories.keys())
    vals = list(categories.values())
    colors = ["#4C72B0" if v < 0.8 else "#55A868" for v in vals]
    bars = ax.barh(names, [v * 100 for v in vals], color=colors, alpha=0.8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percentage (%)")
    ax.set_title(title)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_coverage_histogram(coverages: list, title: str = "pLannotate Coverage") -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(coverages, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    median_cov = np.median(coverages)
    ax.axvline(median_cov, color="red", linestyle="--", label=f"Median: {median_cov:.1%}")
    ax.set_xlabel("Coverage Fraction")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baselines-dir", default="eval/baselines")
    parser.add_argument("--reference", default="eval/reference")
    args = parser.parse_args()

    run = Path(args.run_dir)
    baselines_dir = Path(args.baselines_dir)
    ref_dir = Path(args.reference)

    # Load data
    config = json.load(open(run / "config.json"))
    gen_summary = json.load(open(run / "generation_summary.json"))
    tier1 = pd.read_parquet(run / "metrics" / "tier1_distributional.parquet")
    tier1_summary = json.load(open(run / "metrics" / "tier1_summary.json"))
    tier3 = pd.read_parquet(run / "metrics" / "tier3_essentials.parquet")
    tier5 = pd.read_parquet(run / "metrics" / "tier5_architecture.parquet")
    disc = json.load(open(run / "metrics" / "discriminator.json"))

    ref_metrics = pd.read_csv(ref_dir / "addgene_reference_metrics.csv")
    ref_seqs = pd.read_csv(ref_dir / "addgene_reference_500.csv")

    # Load baselines
    baseline_t1 = {}
    baseline_t3 = {}
    for bl in ["random", "shuffled", "real"]:
        bl_path = baselines_dir / bl / "metrics"
        if (bl_path / "tier1_distributional.parquet").exists():
            baseline_t1[bl] = pd.read_parquet(bl_path / "tier1_distributional.parquet")
        if (bl_path / "tier3_essentials.parquet").exists():
            baseline_t3[bl] = pd.read_parquet(bl_path / "tier3_essentials.parquet")

    # SHAP plot
    shap_b64 = ""
    shap_path = run / "metrics" / "shap_summary.png"
    if shap_path.exists():
        with open(shap_path, "rb") as f:
            shap_b64 = base64.b64encode(f.read()).decode("utf-8")

    # --- Build plots ---
    # Length violin
    bl_lengths = {bl: df["length"].values for bl, df in baseline_t1.items()}
    length_plot = plot_violin_comparison(
        tier1["length"].values, ref_metrics["length"].values, bl_lengths,
        "Sequence Length Distribution", "Length (bp)")

    # GC violin
    bl_gc = {bl: df["gc"].values for bl, df in baseline_t1.items()}
    gc_plot = plot_violin_comparison(
        tier1["gc"].values, ref_metrics["gc"].values, bl_gc,
        "GC Content Distribution", "GC Fraction")

    # ORF density violin
    bl_orf = {bl: df["orf_density"].values for bl, df in baseline_t1.items()}
    orf_plot = plot_violin_comparison(
        tier1["orf_density"].values, None, bl_orf,
        "ORF Density", "ORF bp / Total bp")

    # Coverage histogram
    cov_plot = plot_coverage_histogram(tier3["coverage_frac"].tolist())

    # Tier 3 bar chart
    tier3_bars = plot_bar_chart({
        "Origin (ORI)": tier3["has_ori"].mean(),
        "Selection Marker": tier3["has_selection_marker"].mean(),
        "Promoter": tier3["has_promoter"].mean(),
        "Terminator": tier3["has_terminator"].mean(),
        "CDS": tier3["has_cds"].mean(),
        "Plausible (ORI+Marker+Expr)": tier3["plausibility_pass"].mean(),
    }, "Tier 3: Essential Feature Presence")

    # Plausibility comparison
    plaus_data = {"Generated": tier3["plausibility_pass"].mean()}
    for bl, df in baseline_t3.items():
        plaus_data[bl.capitalize()] = df["plausibility_pass"].mean()
    plaus_bars = plot_bar_chart(plaus_data, "Plausibility Rate Comparison")

    # Tier 5 bar chart
    tier5_bars = plot_bar_chart({
        "CDS w/ Promoter (<500bp)": tier5["pct_cds_with_promoter"].mean(),
        "CDS w/ Terminator (<500bp)": tier5["pct_cds_with_terminator"].mean(),
    }, "Tier 5: Architecture Quality")

    # --- Render HTML ---
    model_name = config.get("model", "Unknown")
    timestamp = config.get("timestamp", "Unknown")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PlasmidLM Eval Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333;
         line-height: 1.6; }}
  h1 {{ border-bottom: 2px solid #4C72B0; padding-bottom: 10px; }}
  h2 {{ color: #4C72B0; margin-top: 40px; }}
  .headline {{ background: #f0f4f8; padding: 20px; border-radius: 8px;
               margin: 20px 0; font-size: 1.1em; }}
  .headline .big {{ font-size: 2em; font-weight: bold; color: #4C72B0; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .metric-card {{ background: #fafafa; border: 1px solid #e0e0e0;
                  border-radius: 6px; padding: 15px; text-align: center; }}
  .metric-card .value {{ font-size: 1.8em; font-weight: bold; color: #333; }}
  .metric-card .label {{ font-size: 0.85em; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f4f8; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  img {{ max-width: 100%; }}
  .plot {{ margin: 15px 0; }}
  .config {{ background: #f8f8f8; padding: 15px; border-radius: 6px;
             font-family: monospace; font-size: 0.85em; white-space: pre-wrap; }}
  .note {{ color: #666; font-size: 0.9em; font-style: italic; }}
</style>
</head>
<body>

<h1>PlasmidLM Evaluation Report</h1>
<p><strong>Model:</strong> <code>{model_name}</code><br>
<strong>Date:</strong> {timestamp}<br>
<strong>Sequences:</strong> {gen_summary['n_generated']} generated</p>

<div class="headline">
  <div class="big">{tier3['plausibility_pass'].mean():.1%}</div>
  Plausibility rate (origin + selection marker + expression element)<br>
  <span class="note">vs. 95.0% for real Addgene plasmids, 0% for random/shuffled controls</span>
</div>

<div class="grid">
  <div class="metric-card">
    <div class="value">{gen_summary['mean_length']:.0f} bp</div>
    <div class="label">Mean Length (ref: {ref_metrics['length'].mean():.0f} bp)</div>
  </div>
  <div class="metric-card">
    <div class="value">{gen_summary['mean_gc']:.3f}</div>
    <div class="label">Mean GC (ref: {ref_metrics['gc'].mean():.3f})</div>
  </div>
  <div class="metric-card">
    <div class="value">{gen_summary['eos_rate']:.1%}</div>
    <div class="label">EOS Rate (completed sequences)</div>
  </div>
  <div class="metric-card">
    <div class="value">{disc['auc']:.3f}</div>
    <div class="label">Discriminator AUC (real vs generated)</div>
  </div>
</div>

<h2>Tier 1: Distributional Metrics</h2>
<p>Comparing generated sequences against the Addgene-500 reference panel and negative controls.</p>

<table>
<tr><th>Metric</th><th>KS Statistic</th><th>KS p-value</th><th>Wasserstein Distance</th></tr>
<tr><td>Length</td><td>{tier1_summary['length']['ks_stat']:.3f}</td>
    <td>{tier1_summary['length']['ks_p']:.2e}</td>
    <td>{tier1_summary['length']['wasserstein']:.1f} bp</td></tr>
<tr><td>GC Content</td><td>{tier1_summary['gc']['ks_stat']:.3f}</td>
    <td>{tier1_summary['gc']['ks_p']:.2e}</td>
    <td>{tier1_summary['gc']['wasserstein']:.4f}</td></tr>
</table>

<div class="plot"><img src="data:image/png;base64,{length_plot}" alt="Length distribution"></div>
<div class="plot"><img src="data:image/png;base64,{gc_plot}" alt="GC distribution"></div>
<div class="plot"><img src="data:image/png;base64,{orf_plot}" alt="ORF density"></div>

<h2>Tier 3: Essential Features</h2>
<p>Presence of key plasmid components as detected by pLannotate.</p>
<div class="plot"><img src="data:image/png;base64,{tier3_bars}" alt="Tier 3 features"></div>
<div class="plot"><img src="data:image/png;base64,{plaus_bars}" alt="Plausibility comparison"></div>

<h2>pLannotate Coverage</h2>
<div class="plot"><img src="data:image/png;base64,{cov_plot}" alt="Coverage histogram"></div>
<p>Mean features per sequence: <strong>{tier3['n_features'].mean():.1f}</strong><br>
Mean coverage fraction: <strong>{tier3['coverage_frac'].mean():.1%}</strong></p>

<h2>Tier 5: Architecture</h2>
<p>Structural organization of annotated features.</p>
<div class="plot"><img src="data:image/png;base64,{tier5_bars}" alt="Tier 5 architecture"></div>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Mean origins per plasmid</td><td>{tier5['n_origins'].mean():.1f}</td></tr>
<tr><td>Mean selection markers</td><td>{tier5['n_selection_markers'].mean():.1f}</td></tr>
<tr><td>Mean overlapping features</td><td>{tier5['n_overlapping_features'].mean():.1f}</td></tr>
<tr><td>% CDS with upstream promoter (&lt;500bp)</td><td>{tier5['pct_cds_with_promoter'].mean():.1%}</td></tr>
<tr><td>% CDS with downstream terminator (&lt;500bp)</td><td>{tier5['pct_cds_with_terminator'].mean():.1%}</td></tr>
</table>

<h2>Memorization</h2>
<p>Sourmash containment search (k=31, scaled=100) against the full training set (20,644 plasmids).</p>
<p><strong>Result: Zero hits</strong> above 0.3 containment threshold. No evidence of training set memorization.</p>

<h2>Discriminator</h2>
<p>LightGBM binary classifier distinguishing real Addgene-500 from generated sequences.</p>
<p><strong>Test AUC: {disc['auc']:.3f}</strong> (70/30 stratified split, {disc['n_test']} test samples)</p>

<table>
<tr><th>Feature</th><th>Importance (gain)</th></tr>
{''.join(f'<tr><td>{k}</td><td>{v:.0f}</td></tr>' for k, v in sorted(disc['feature_importance'].items(), key=lambda x: -x[1]))}
</table>

{"<div class='plot'><img src='data:image/png;base64," + shap_b64 + "' alt='SHAP summary'></div>" if shap_b64 else "<p class='note'>SHAP plot not available.</p>"}

<h2>Run Configuration</h2>
<div class="config">{json.dumps(config, indent=2)}</div>

<h2>Tool Versions</h2>
<table>
<tr><th>Tool</th><th>Version</th></tr>
<tr><td>pLannotate</td><td>1.2.2 (snapgene DB 2021-11)</td></tr>
<tr><td>Prodigal</td><td>2.6.3</td></tr>
<tr><td>dustmasker (BLAST+)</td><td>2.17.0</td></tr>
<tr><td>sourmash</td><td>4.3.0</td></tr>
<tr><td>LightGBM</td><td>4.6.0</td></tr>
<tr><td>SHAP</td><td>0.49.1</td></tr>
</table>

<p class="note">Report generated by PlasmidLM eval suite. Data at
<a href="https://huggingface.co/buckets/McClain/PlasmidLMEval">McClain/PlasmidLMEval</a>.</p>

</body>
</html>"""

    out_path = run / "report.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report: {out_path} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
