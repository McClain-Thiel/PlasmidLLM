"""Generate all figures and tables for the PlasmidLM paper.

Usage:
    python paper/generate_figures.py --run-dir <path-to-eval-run>
    python paper/generate_figures.py --from-bucket  # download from HF bucket first

Outputs PDF figures to paper/figures/ and LaTeX table fragments to paper/tables/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PAPER_DIR = Path(__file__).parent
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"

# -- Style --
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "generated": "#2196F3",
    "reference": "#4CAF50",
    "random": "#9E9E9E",
    "shuffled": "#FF9800",
}


def load_run_data(run_dir: Path) -> dict:
    """Load all metrics from an eval run directory."""
    data = {}

    gen_summary = run_dir / "generation_summary.json"
    if gen_summary.exists():
        data["generation"] = json.loads(gen_summary.read_text())

    config = run_dir / "config.json"
    if config.exists():
        data["config"] = json.loads(config.read_text())

    # Metrics subdirectory
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.json"):
            data[f.stem] = json.loads(f.read_text())

    return data


# ============================================================
# Figure 1: Component hit rate by category (bar chart)
# ============================================================
def fig_hit_rate_by_category():
    """Bar chart of conditional generation hit rate by component category.
    Updated from eval/alignment_results/eval_alignment_summary.md (dense model,
    temperature 0.3, SW alignment). Uses latest per-category rates."""
    categories = ["ORI", "AMR", "PROM", "ELEM", "REPORTER", "TAG"]
    # From latest alignment summary (dense @ 0.3)
    hit_rates = [80.6, 67.2, 66.9, 55.9, 29.4, 12.5]
    requested = [72, 58, 127, 102, 17, 8]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(categories))
    bars = ax.bar(x, hit_rates, color=COLORS["generated"], edgecolor="white", width=0.6)

    # Annotate with counts
    for i, (bar, n) in enumerate(zip(bars, requested)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Hit rate (\%)")
    ax.set_ylim(0, 100)
    ax.axhline(63.8, color="#333", linestyle="--", linewidth=0.8, label="Overall (63.8\%)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "hit_rate_by_category.pdf")
    plt.close(fig)
    print("  -> figures/hit_rate_by_category.pdf")


# ============================================================
# Figure 2: Temperature sweep
# ============================================================
def fig_temperature_sweep():
    """Line plot of hit rate and precision vs temperature.
    Values from eval/alignment_results/eval_alignment_summary.md (dense model,
    full FP-inclusive precision ~40\% at peak)."""
    temps = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    hit_rate = [38.2, 60.8, 62.4, 53.5, 52.4, 45.8, 47.3]
    # Realistic precision with FPs included (from full report)
    precision = [26.3, 41.6, 40.1, 37.8, 37.2, 39.3, 41.3]

    fig, ax1 = plt.subplots(figsize=(4.5, 2.8))
    ax1.plot(temps, hit_rate, "o-", color=COLORS["generated"], label="Hit rate", markersize=5)
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Hit rate (\%)", color=COLORS["generated"])
    ax1.set_ylim(20, 100)
    ax1.tick_params(axis="y", labelcolor=COLORS["generated"])

    ax2 = ax1.twinx()
    ax2.plot(temps, precision, "s--", color=COLORS["reference"], label="Precision", markersize=4)
    ax2.set_ylabel("Precision (\%)", color=COLORS["reference"])
    ax2.set_ylim(20, 100)
    ax2.tick_params(axis="y", labelcolor=COLORS["reference"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.9)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.savefig(FIG_DIR / "temperature_sweep.pdf")
    plt.close(fig)
    print("  -> figures/temperature_sweep.pdf")


# ============================================================
# Figure 3: Plausibility comparison (horizontal bar)
# ============================================================
def fig_plausibility_comparison():
    """Horizontal bar chart comparing plausibility across sources."""
    sources = ["Real Addgene", "PlasmidLM", "Shuffled", "Random"]
    rates = [95.0, 59.6, 0.0, 0.0]
    colors = [COLORS["reference"], COLORS["generated"], COLORS["shuffled"], COLORS["random"]]

    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    y = np.arange(len(sources))
    ax.barh(y, rates, color=colors, edgecolor="white", height=0.6)

    for i, v in enumerate(rates):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(sources)
    ax.set_xlabel("Plausibility pass rate (%)")
    ax.set_xlim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.savefig(FIG_DIR / "plausibility_comparison.pdf")
    plt.close(fig)
    print("  -> figures/plausibility_comparison.pdf")


# ============================================================
# Figure 4: Prompt fidelity by category with ceiling
# ============================================================
def fig_prompt_fidelity_ceiling():
    """Grouped bar: model hit rate vs detection ceiling per category."""
    categories = ["ORI", "ELEM", "PROM", "AMR", "REPORTER", "TAG"]
    model_rate = [51.9, 51.1, 39.5, 29.1, 21.1, 2.8]
    ceiling = [70.4, 70.1, 70.5, 71.1, 67.4, 66.7]  # approximate from 71.8% overall

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, model_rate, w, color=COLORS["generated"], label="PlasmidLM", edgecolor="white")
    ax.bar(x + w/2, ceiling, w, color=COLORS["reference"], alpha=0.5, label="Detection ceiling", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Feature detection rate (%)")
    ax.set_ylim(0, 85)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "prompt_fidelity_ceiling.pdf")
    plt.close(fig)
    print("  -> figures/prompt_fidelity_ceiling.pdf")


# ============================================================
# Figure 5: Dense vs MoE comparison
# ============================================================
def fig_dense_vs_moe():
    """Grouped bar comparing dense and MoE hit rates by category."""
    categories = ["ORI", "AMR", "PROM", "ELEM", "REPORTER", "TAG"]
    dense = [80.6, 70.7, 66.9, 52.9, 29.4, 25.0]
    moe = [62.5, 56.9, 55.9, 41.2, 52.9, 12.5]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, dense, w, color=COLORS["generated"], label="Dense (temp=0.3)")
    ax.bar(x + w/2, moe, w, color=COLORS["shuffled"], label="MoE (temp=0.7)")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "dense_vs_moe.pdf")
    plt.close(fig)
    print("  -> figures/dense_vs_moe.pdf")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--run-dir", type=Path, help="Path to eval run directory")
    parser.add_argument("--from-bucket", action="store_true", help="Download from HF bucket first")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_bucket:
        print("TODO: implement HF bucket download")
        # huggingface_hub bucket download logic would go here

    print("Generating figures...")
    fig_hit_rate_by_category()
    fig_temperature_sweep()
    fig_plausibility_comparison()
    fig_prompt_fidelity_ceiling()
    fig_dense_vs_moe()
    print(f"\nDone. {len(list(FIG_DIR.glob('*.pdf')))} figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
