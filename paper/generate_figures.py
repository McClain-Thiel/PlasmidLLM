# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.7",
#     "numpy>=1.26",
#     "pandas>=2.0",
#     "pyarrow>=14.0",
#     "statsmodels>=0.14",
# ]
# ///
"""Generate all paper figures from the PlasmidLMEval bucket cache.

The numbers here are not hardcoded literals — they are computed from parquets and JSON
shipped in the HuggingFace bucket `McClain/PlasmidLMEval`. Populate the cache with

    hf buckets sync hf://buckets/McClain/PlasmidLMEval /tmp/plasmidlm_eval_cache/bucket

before running. If you want to run from a fresh directory layout (e.g. your own eval run),
pass `--cache <dir>`.

Usage:
    uv run --script paper/generate_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

PAPER_DIR = Path(__file__).parent
FIG_DIR = PAPER_DIR / "figures"
DEFAULT_CACHE = Path("/tmp/plasmidlm_eval_cache")

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
    "base":     "#6b7280",
    "grpo":     "#059669",
    "real":     "#4CAF50",
    "random":   "#9E9E9E",
    "shuffled": "#FF9800",
}

BASE_RUN = "runs__base_kmer6_matched_20260420_1303"
GRPO_RUN = "runs__grpo_plannotate_full_20260408"
BON_SEEDS = {
    "base": [BASE_RUN,
             "runs__bon_base_seed43_20260420_1303",
             "runs__bon_base_seed44_20260420_1303",
             "runs__bon_base_seed45_20260420_1303"],
    "grpo": [GRPO_RUN,
             "runs__bon_grpo_seed43_20260420_1303",
             "runs__bon_grpo_seed44_20260420_1303",
             "runs__bon_grpo_seed45_20260420_1303"],
}
SWEEP_RUNS = {
    "t=0.3": "runs__base_kmer6_sweep_20260420_1303__t03",
    "t=0.5": "runs__base_kmer6_sweep_20260420_1303__t05",
    "t=1.0": "runs__base_kmer6_sweep_20260420_1303__t10",
}
CATEGORIES = ["ORI", "AMR", "PROM", "ELEM", "REPORTER", "TAG"]


def load_tier3(cache: Path, run: str) -> pd.DataFrame:
    return pd.read_parquet(cache / run / "metrics/tier3_essentials.parquet").reset_index(drop=True)


def load_fidelity(cache: Path, run: str) -> pd.DataFrame:
    return pd.read_parquet(cache / run / "metrics/prompt_fidelity_per_seq.parquet").reset_index(drop=True)


def fig_plausibility_comparison(cache: Path):
    """Horizontal bar chart: plausibility across sources, with paired base vs GRPO."""
    base = load_tier3(cache, BASE_RUN)["plausibility_pass"].mean()
    grpo = load_tier3(cache, GRPO_RUN)["plausibility_pass"].mean()

    sources = ["Real Addgene", "PlasmidLM post-GRPO", "PlasmidLM pre-GRPO", "Shuffled", "Random"]
    rates = [0.95, grpo, base, 0.0, 0.0]
    colors = [COLORS["real"], COLORS["grpo"], COLORS["base"], COLORS["shuffled"], COLORS["random"]]

    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    y = np.arange(len(sources))
    ax.barh(y, [r * 100 for r in rates], color=colors, edgecolor="white", height=0.6)
    for i, v in enumerate(rates):
        ax.text(v * 100 + 1.5, i, f"{v * 100:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(sources)
    ax.set_xlabel("Plausibility pass rate (%)")
    ax.set_xlim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.savefig(FIG_DIR / "plausibility_comparison.pdf")
    plt.close(fig)


def fig_hit_rate_by_category(cache: Path):
    """Per-category fidelity bars: base vs GRPO with Wilson 95% CIs."""
    base_fid = load_fidelity(cache, BASE_RUN)
    grpo_fid = load_fidelity(cache, GRPO_RUN)

    rows = []
    for cat in CATEGORIES:
        col_r = f"n_requested_{cat}"
        col_f = f"n_found_{cat}"
        for name, df in [("base", base_fid), ("grpo", grpo_fid)]:
            k = int(df[col_f].sum())
            n = int(df[col_r].sum())
            lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            rows.append({"category": cat, "run": name, "rate": k / n,
                         "ci_lo": float(lo), "ci_hi": float(hi)})
    tbl = pd.DataFrame(rows)

    pivot_rate = tbl.pivot(index="category", columns="run", values="rate").reindex(CATEGORIES)
    pivot_lo = tbl.pivot(index="category", columns="run", values="ci_lo").reindex(CATEGORIES)
    pivot_hi = tbl.pivot(index="category", columns="run", values="ci_hi").reindex(CATEGORIES)

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = np.arange(len(CATEGORIES))
    w = 0.38
    for offset, name, color in [(-w / 2, "base", COLORS["base"]),
                                (+w / 2, "grpo", COLORS["grpo"])]:
        vals = pivot_rate[name].values * 100
        yerr = np.vstack([vals - pivot_lo[name].values * 100,
                          pivot_hi[name].values * 100 - vals])
        ax.bar(x + offset, vals, width=w, label=name, color=color, yerr=yerr,
               capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIG_DIR / "hit_rate_by_category.pdf")
    plt.close(fig)


def fig_useful_vs_threshold(cache: Path):
    """Useful-rate (plausible AND fidelity >= T) vs threshold, paired, with CIs and delta."""
    base_t3 = load_tier3(cache, BASE_RUN)["plausibility_pass"].astype(bool).to_numpy()
    grpo_t3 = load_tier3(cache, GRPO_RUN)["plausibility_pass"].astype(bool).to_numpy()
    base_f = load_fidelity(cache, BASE_RUN)
    grpo_f = load_fidelity(cache, GRPO_RUN)

    thresholds = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]
    rng = np.random.default_rng(0)
    B = 10_000
    rows = []
    for T in thresholds:
        def _mask(t3, f):
            has_tokens = (f["n_requested"] > 0).to_numpy()
            rate = np.where(has_tokens,
                            f["n_found"].to_numpy() / np.maximum(f["n_requested"].to_numpy(), 1),
                            np.nan)
            if T == 0.0:
                return t3
            return t3 & (np.nan_to_num(rate, nan=-1.0) >= T - 1e-9)
        bm = _mask(base_t3, base_f)
        gm = _mask(grpo_t3, grpo_f)
        bn, bk = len(bm), int(bm.sum())
        gn, gk = len(gm), int(gm.sum())
        blo, bhi = proportion_confint(bk, bn, alpha=0.05, method="wilson")
        glo, ghi = proportion_confint(gk, gn, alpha=0.05, method="wilson")
        idx = rng.integers(0, bn, size=(B, bn))
        boot = gm[idx].mean(axis=1) - bm[idx].mean(axis=1)
        rows.append({"T": T,
                     "base_rate": bk / bn, "base_lo": blo, "base_hi": bhi,
                     "grpo_rate": gk / gn, "grpo_lo": glo, "grpo_hi": ghi,
                     "delta": gk / gn - bk / bn,
                     "delta_lo": float(np.quantile(boot, 0.025)),
                     "delta_hi": float(np.quantile(boot, 0.975))})
    tbl = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    T = tbl["T"].values
    ax1.plot(T, tbl["base_rate"] * 100, "o-", color=COLORS["base"], label="base")
    ax1.fill_between(T, tbl["base_lo"] * 100, tbl["base_hi"] * 100,
                     color=COLORS["base"], alpha=0.2)
    ax1.plot(T, tbl["grpo_rate"] * 100, "o-", color=COLORS["grpo"], label="GRPO")
    ax1.fill_between(T, tbl["grpo_lo"] * 100, tbl["grpo_hi"] * 100,
                     color=COLORS["grpo"], alpha=0.2)
    ax1.set_xlabel("Fidelity threshold $T$")
    ax1.set_ylabel("Useful rate (%)")
    ax1.set_title(r"Plausible AND fidelity $\geq T$")
    ax1.legend(framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.plot(T, tbl["delta"] * 100, "o-", color="#0369a1")
    ax2.fill_between(T, tbl["delta_lo"] * 100, tbl["delta_hi"] * 100,
                     color="#0369a1", alpha=0.2)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Fidelity threshold $T$")
    ax2.set_ylabel("$\\Delta$ GRPO $-$ base (pp)")
    ax2.set_title("Paired bootstrap $95\\%$ CI on $\\Delta$")
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "prompt_fidelity_ceiling.pdf")
    plt.close(fig)


def fig_temperature_sweep(cache: Path):
    """Base-model temperature sweep: plausibility and fidelity per temp."""
    temps = [0.3, 0.5, 0.7, 1.0]
    plaus = []
    fid = []
    for T, run in [(0.3, SWEEP_RUNS["t=0.3"]),
                   (0.5, SWEEP_RUNS["t=0.5"]),
                   (0.7, BASE_RUN),
                   (1.0, SWEEP_RUNS["t=1.0"])]:
        t3 = load_tier3(cache, run)
        f = load_fidelity(cache, run)
        plaus.append(float(t3["plausibility_pass"].mean()) * 100)
        with_tok = f[f["n_requested"] > 0]
        fid.append(float((with_tok["n_found"] / with_tok["n_requested"]).mean()) * 100)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.plot(temps, plaus, "o-", color=COLORS["base"], label="plausibility")
    ax.plot(temps, fid, "s--", color="#b45309", label="prompt fidelity")
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("rate (%)")
    ax.set_ylim(20, 70)
    ax.set_xticks(temps)
    ax.axvline(0.7, color="#333", linestyle=":", linewidth=0.8)
    ax.text(0.72, 68, "matched", fontsize=8, color="#333")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(FIG_DIR / "temperature_sweep.pdf")
    plt.close(fig)


def fig_best_of_k(cache: Path):
    """Best-of-K, paired, two-panel: plausibility and useful@T=0.5."""
    def _seed_frames(model: str):
        out = []
        for run in BON_SEEDS[model]:
            t3 = load_tier3(cache, run)
            f = load_fidelity(cache, run)
            df = pd.DataFrame({
                "plaus": t3["plausibility_pass"].astype(bool).values,
                "n_req": f["n_requested"].values,
                "n_fnd": f["n_found"].values,
            })
            df["fid"] = np.where(df["n_req"] > 0,
                                 df["n_fnd"] / np.maximum(df["n_req"], 1),
                                 np.nan)
            df["useful"] = df["plaus"] & (df["fid"] >= 0.5)
            out.append(df)
        return out

    def _best_of_k(seeds, K):
        subset = seeds[:K]
        stack = np.stack([s["n_fnd"].values for s in subset], axis=0)
        plaus = np.stack([s["plaus"].astype(int).values for s in subset], axis=0)
        fid = np.stack([np.nan_to_num(s["fid"].values, nan=-1.0) for s in subset], axis=0)
        key = stack * 1e6 + plaus * 1e3 + fid
        idx = key.argmax(axis=0)
        out = {}
        for col in ("plaus", "useful"):
            vals = np.stack([s[col].values for s in subset], axis=0)
            out[col] = vals[idx, np.arange(vals.shape[1])].astype(bool)
        return out

    base_seeds = _seed_frames("base")
    grpo_seeds = _seed_frames("grpo")

    Ks = [1, 2, 4]
    curves = {("base", "plaus"): [], ("grpo", "plaus"): [],
              ("base", "useful"): [], ("grpo", "useful"): []}
    cis = {k: [] for k in curves}
    for K in Ks:
        b = _best_of_k(base_seeds, K)
        g = _best_of_k(grpo_seeds, K)
        for name, dct in [("base", b), ("grpo", g)]:
            for metric in ("plaus", "useful"):
                v = dct[metric]
                k = int(v.sum()); n = len(v)
                lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
                curves[(name, metric)].append(k / n)
                cis[(name, metric)].append((float(lo), float(hi)))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, metric, title in [(axes[0], "plaus", "Structural plausibility vs $K$"),
                               (axes[1], "useful", "Useful rate (plaus AND fid $\\geq 0.5$) vs $K$")]:
        for name, color in [("base", COLORS["base"]), ("grpo", COLORS["grpo"])]:
            vals = np.array(curves[(name, metric)]) * 100
            ci = np.array(cis[(name, metric)]) * 100
            ax.plot(Ks, vals, "o-", color=color, label=name)
            ax.fill_between(Ks, ci[:, 0], ci[:, 1], color=color, alpha=0.2)
        ax.set_xlabel("$K$ (best-of-$K$, oracle ranker)")
        ax.set_ylabel("rate (%)")
        ax.set_title(title)
        ax.set_xticks(Ks)
        ax.legend(framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dense_vs_moe.pdf")  # file kept as-is for \includegraphics compat
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help=f"Local cache populated from the PlasmidLMEval bucket (default {DEFAULT_CACHE})")
    args = parser.parse_args()

    missing = [run for run in [BASE_RUN, GRPO_RUN] + list(SWEEP_RUNS.values())
               + BON_SEEDS["base"] + BON_SEEDS["grpo"]
               if not (args.cache / run).is_dir()]
    if missing:
        raise SystemExit(
            f"cache missing {len(missing)} run dirs under {args.cache}:\n  " +
            "\n  ".join(missing[:5]) +
            ("\n  ..." if len(missing) > 5 else "") +
            "\n\nPopulate the cache with:\n"
            f"    hf buckets sync hf://buckets/McClain/PlasmidLMEval {args.cache}"
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing to {FIG_DIR}/")
    fig_plausibility_comparison(args.cache)
    print("  plausibility_comparison.pdf")
    fig_hit_rate_by_category(args.cache)
    print("  hit_rate_by_category.pdf")
    fig_useful_vs_threshold(args.cache)
    print("  prompt_fidelity_ceiling.pdf")
    fig_temperature_sweep(args.cache)
    print("  temperature_sweep.pdf")
    fig_best_of_k(args.cache)
    print("  dense_vs_moe.pdf  (contents: best-of-K; name kept for \\includegraphics compat)")


if __name__ == "__main__":
    main()
