#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas==2.3.3",
#     "pyarrow>=14.0",
#     "numpy==2.2.6",
#     "scipy>=1.11",
#     "statsmodels==0.14.6",
#     "matplotlib==3.10.8",
# ]
# ///
"""Generate the soft-token conditioning figures used in the paper.

Produces three figures (both PNG and PDF), consumed directly by
`paper/main.tex` and referenced in the accompanying notebook:

    soft_token_forest          — effect-size forest plot across all soft values
    soft_token_vec_heatmap     — 5×5 VEC_ heatmap, base vs GRPO
    soft_token_copy_buckets    — COPY_HIGH/LOW 4-bucket breakdown

All figures come from the same cached paired runs as
`notebooks/soft_token_conditioning.py`:

    /tmp/plasmidlm_eval_cache/runs__base_kmer6_matched_20260420_1303
    /tmp/plasmidlm_eval_cache/runs__grpo_plannotate_full_20260408

Usage:
    uv run --script eval/scripts/make_soft_token_figures.py \
        --out eval/figures/soft_token
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

CACHE = Path("/tmp/plasmidlm_eval_cache")
BASE_DIR = CACHE / "runs__base_kmer6_matched_20260420_1303"
GRPO_DIR = CACHE / "runs__grpo_plannotate_full_20260408"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "data" / "motif_registry_combined.parquet"
)

SOFT_TO_HARD = {
    "COPY": {
        "HIGH": {"ORI_COLE1", "ORI_RSF"},
        "LOW":  {"ORI_P15A", "ORI_PSC101"},
    },
    "VEC": {
        "MAMMALIAN":  {"PROM_CMV", "PROM_EF1A", "PROM_CAG",
                       "ELEM_POLYA_BGH", "ELEM_POLYA_SV40", "ELEM_CMV_ENHANCER"},
        "BACTERIAL":  {"PROM_T7", "PROM_T5", "PROM_LAC"},
        "LENTIVIRAL": {"ELEM_LTR_3", "ELEM_LTR_5", "ELEM_PSI", "ELEM_CPPT", "ELEM_WPRE"},
        "CRISPR":     {"ELEM_GRNA_SCAFFOLD", "ELEM_TRACRRNA", "PROM_U6"},
        "AAV":        {"ELEM_AAV_ITR"},
    },
    "SP": {
        "HUMAN": {"PROM_CMV", "PROM_EF1A", "ELEM_POLYA_BGH"},
        "ECOLI": {"PROM_T7", "PROM_T5", "PROM_LAC"},
        "YEAST": {"ORI_2MU"},
    },
    "BB": {
        "PUC19":            {"AMR_AMPICILLIN", "ORI_COLE1"},
        "PCDNA3":           {"PROM_CMV", "ELEM_POLYA_BGH", "AMR_NEOMYCIN", "ORI_SV40"},
        "PEGFP_C1":         {"REPORTER_EGFP", "PROM_CMV", "AMR_KANAMYCIN"},
        "PEGFP_N1":         {"REPORTER_EGFP", "PROM_CMV", "AMR_KANAMYCIN"},
        "PAAV":             {"ELEM_AAV_ITR"},
        "PAAV2":            {"ELEM_AAV_ITR"},
    },
}
UNCLASSIFIED_FOR = {"COPY": {"ORI_F1", "ORI_SV40", "ORI_2MU"}}
SOFT_RE = re.compile(r"<(BB|COPY|SP|VEC)_([^>]+)>")


def load_run(run_dir: Path) -> dict:
    gens = pd.read_parquet(run_dir / "generations.parquet").reset_index(drop=True)
    with open(run_dir / "plannotate_results.json") as f:
        pn = json.load(f)
    sseqids = {
        e["seq_id"]: {
            f["sseqid"] for f in e.get("features", [])
            if isinstance(f.get("sseqid"), str) and f["sseqid"]
        }
        for e in pn
    }
    by_row = [sseqids.get(f"gen_{i}", set()) for i in range(len(gens))]
    return {"gens": gens, "sseqids_by_row": by_row}


def load_registry(path: Path) -> dict[str, set[str]]:
    df = pd.read_parquet(path, columns=["token", "sseqid"])
    m: dict[str, set[str]] = defaultdict(set)
    for tok, sid in zip(df["token"], df["sseqid"]):
        if isinstance(tok, str) and isinstance(sid, str):
            m[tok.strip("<>")].add(sid)
    return dict(m)


def extract_tag(prompts: list[str], cat: str) -> list[str | None]:
    out: list[str | None] = []
    for p in prompts:
        found: str | None = None
        for c, v in SOFT_RE.findall(p):
            if c == cat and found is None:
                found = v
        out.append(found)
    return out


def resolve_probes(token_sseqids: dict[str, set[str]]) -> dict[str, dict[str, set[str]]]:
    return {
        cat: {
            val: set().union(*(token_sseqids.get(t, set()) for t in toks))
            for val, toks in by_val.items()
        }
        for cat, by_val in SOFT_TO_HARD.items()
    }


def hit(run: dict, mask: np.ndarray, probe: set[str]) -> tuple[int, int]:
    n = int(mask.sum())
    if n == 0:
        return 0, 0
    k = sum(
        1 for i, m in enumerate(mask)
        if m and run["sseqids_by_row"][i] & probe
    )
    return k, n


def compute_discrimination(
    base: dict, grpo: dict, prompts: list[str],
    probes: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    """For each (cat, val) with a probe: P(hit) on prompts tagged val vs prompts
    tagged other values in the same category. Returns long-form with Wilson CIs
    and Fisher p, one row per (run, cat, val)."""
    rows = []
    for cat, by_val in probes.items():
        tags = np.array(extract_tag(prompts, cat))
        for val, probe in by_val.items():
            if not probe:
                continue
            mask_A = tags == val
            mask_B = (tags != val) & (tags != None)  # noqa: E711
            for run_name, run in [("base", base), ("grpo", grpo)]:
                k_A, n_A = hit(run, mask_A, probe)
                k_B, n_B = hit(run, mask_B, probe)
                if n_A < 3 or n_B < 3:
                    continue
                r_A = k_A / n_A
                r_B = k_B / n_B
                lo_A, hi_A = proportion_confint(k_A, n_A, alpha=0.05, method="wilson")
                lo_B, hi_B = proportion_confint(k_B, n_B, alpha=0.05, method="wilson")
                _odds, p = fisher_exact([[k_A, n_A - k_A], [k_B, n_B - k_B]])
                rows.append({
                    "run": run_name, "cat": cat, "val": val,
                    "n_A": n_A, "k_A": k_A, "rate_A": r_A,
                    "ci_low_A": float(lo_A), "ci_high_A": float(hi_A),
                    "n_B": n_B, "rate_B": r_B,
                    "ci_low_B": float(lo_B), "ci_high_B": float(hi_B),
                    "delta": r_A - r_B,
                    "fisher_p": float(p),
                })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Figure 1 — Forest plot of effect sizes
# ──────────────────────────────────────────────────────────────────────────

def plot_forest(df: pd.DataFrame, out_dir: Path) -> None:
    order = [
        ("VEC", "LENTIVIRAL"), ("VEC", "AAV"), ("VEC", "CRISPR"),
        ("VEC", "MAMMALIAN"), ("VEC", "BACTERIAL"),
        ("SP", "HUMAN"), ("SP", "ECOLI"), ("SP", "YEAST"),
        ("COPY", "HIGH"), ("COPY", "LOW"),
        ("BB", "PAAV"), ("BB", "PAAV2"), ("BB", "PEGFP_C1"), ("BB", "PEGFP_N1"),
        ("BB", "PCDNA3"), ("BB", "PUC19"),
    ]
    rows_b, rows_g = [], []
    for cat, val in order:
        hit_b = df[(df["run"] == "base") & (df["cat"] == cat) & (df["val"] == val)]
        hit_g = df[(df["run"] == "grpo") & (df["cat"] == cat) & (df["val"] == val)]
        if len(hit_b):
            rows_b.append(hit_b.iloc[0])
        else:
            rows_b.append(None)
        if len(hit_g):
            rows_g.append(hit_g.iloc[0])
        else:
            rows_g.append(None)

    # Bootstrap 95% CI on Δ (rate_A − rate_B) via binomial resampling.
    rng = np.random.default_rng(0)
    def _delta_ci(r):
        if r is None:
            return float("nan"), float("nan"), float("nan")
        # Sample A's successes and B's successes independently from their
        # binomials and take the difference.
        draws_A = rng.binomial(r["n_A"], r["rate_A"], size=5000) / max(r["n_A"], 1)
        draws_B = rng.binomial(r["n_B"], r["rate_B"], size=5000) / max(r["n_B"], 1)
        boot = draws_A - draws_B
        return float(r["delta"]), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.0, 0.45 * len(order) + 0.8))
    offset = 0.18
    for rows, color, label, off in [
        (rows_b, "#374151", "base", -offset),
        (rows_g, "#059669", "GRPO", +offset),
    ]:
        deltas, lo_err, hi_err, ys, ps = [], [], [], [], []
        for j, r in enumerate(rows):
            if r is None:
                continue
            d, lo, hi = _delta_ci(r)
            deltas.append(d)
            lo_err.append(d - lo); hi_err.append(hi - d)
            ys.append(y[j] + off)
            ps.append(r["fisher_p"])
        ax.errorbar(
            deltas, ys, xerr=[lo_err, hi_err],
            fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=2.5,
            label=label, markersize=4.5,
        )
        # Annotate significance stars outside the right edge of the CI.
        for d, hi, yy, pv in zip(deltas, hi_err, ys, ps):
            if pv < 0.001:
                s = "***"
            elif pv < 0.01:
                s = "**"
            elif pv < 0.05:
                s = "*"
            else:
                s = ""
            if s:
                ax.text(d + hi + 0.008, yy, s, va="center", ha="left",
                        fontsize=8, color=color)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{cat}_{val}" for cat, val in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ hit-rate  (prompts with tag − prompts without)")
    ax.set_title("Soft-token discrimination — does the probe fire more when\n"
                 "the tag is asked for? (Fisher p: * <0.05, ** <0.01, *** <0.001)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    # Light horizontal separators between categories.
    seps = []
    prev_cat = None
    for j, (cat, _) in enumerate(order):
        if prev_cat is not None and cat != prev_cat:
            seps.append(j - 0.5)
        prev_cat = cat
    for s in seps:
        ax.axhline(s, color="#d1d5db", linewidth=0.6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"soft_token_forest.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 2 — VEC heatmap (base | GRPO)
# ──────────────────────────────────────────────────────────────────────────

def plot_vec_heatmap(base: dict, grpo: dict, prompts: list[str],
                     probes: dict, out_dir: Path) -> None:
    vec_vals = list(probes["VEC"].keys())
    tags = np.array(extract_tag(prompts, "VEC"))

    def _matrix(run):
        mat = np.zeros((len(vec_vals), len(vec_vals)))
        counts = np.zeros(len(vec_vals), dtype=int)
        for i, tag in enumerate(tags):
            if tag not in vec_vals:
                continue
            r = vec_vals.index(tag)
            counts[r] += 1
            for c, col_val in enumerate(vec_vals):
                if run["sseqids_by_row"][i] & probes["VEC"][col_val]:
                    mat[r, c] += 1
        return np.divide(mat, counts[:, None],
                         out=np.zeros_like(mat),
                         where=counts[:, None] > 0), counts

    mat_b, counts = _matrix(base)
    mat_g, _ = _matrix(grpo)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True,
                             gridspec_kw={"top": 0.80, "bottom": 0.22,
                                          "left": 0.12, "right": 0.92,
                                          "wspace": 0.08})
    for ax, mat, title in [(axes[0], mat_b, "base"), (axes[1], mat_g, "GRPO")]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Greens", aspect="auto")
        ax.set_xticks(range(len(vec_vals)))
        ax.set_xticklabels(vec_vals, rotation=30, ha="right")
        ax.set_yticks(range(len(vec_vals)))
        ax.set_yticklabels(
            [f"{v} (n={counts[i]})" for i, v in enumerate(vec_vals)]
        )
        for i in range(len(vec_vals)):
            for j in range(len(vec_vals)):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    color=("white" if mat[i, j] > 0.55 else "black"),
                    fontsize=9,
                )
        ax.set_title(title)
        ax.set_xlabel("probe set")
        # Draw diagonal bounding rect to highlight "correct conditioning" cells.
        for k in range(len(vec_vals)):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1, fill=False,
                edgecolor="black", linewidth=1.2,
            ))
    axes[0].set_ylabel("requested VEC_ tag")
    fig.colorbar(im, ax=axes, shrink=0.7, label="hit rate", pad=0.04)
    fig.suptitle(
        "VEC_ conditioning heatmap — diagonal = requested class\n"
        "(only LENTIVIRAL and AAV show a sharp diagonal effect)",
        fontsize=11, y=0.98,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"soft_token_vec_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 3 — COPY_ buckets (illustrates the COPY_LOW failure)
# ──────────────────────────────────────────────────────────────────────────

def plot_copy_buckets(base: dict, grpo: dict, prompts: list[str],
                      probes: dict, out_dir: Path) -> None:
    tags = np.array(extract_tag(prompts, "COPY"))
    buckets_order = ["correct", "wrong", "unclassified", "no_signal"]
    colors = {"correct": "#059669", "wrong": "#dc2626",
              "unclassified": "#f59e0b", "no_signal": "#6b7280"}

    # Build shares per (run, tag).
    # unclassified = detected an ORI-family sseqid but none of the COPY
    # probes, or hit one of the UNCLASSIFIED_FOR["COPY"] tokens.
    # Resolve those sseqids once.
    family_sseqids_copy = set()
    uncl_tokens = UNCLASSIFIED_FOR["COPY"]
    # We don't re-load the registry here; instead we derive family from the
    # probe sseqids plus uncl sseqids (good enough for this figure — the tiny
    # set of "other" ORIs we miss is folded into no_signal).
    for s in probes["COPY"].values():
        family_sseqids_copy |= s

    # Re-load registry for uncl sseqids (cheap: done once).
    reg = load_registry(DEFAULT_REGISTRY)
    uncl_sseqids = set().union(*(reg.get(t, set()) for t in uncl_tokens))
    family_sseqids_copy |= uncl_sseqids

    def _classify(run):
        labels_by_tag: dict[str, list[str]] = {"HIGH": [], "LOW": []}
        for i, tag in enumerate(tags):
            if tag not in probes["COPY"]:
                continue
            det = run["sseqids_by_row"][i]
            correct_set = probes["COPY"][tag]
            other_tag = "LOW" if tag == "HIGH" else "HIGH"
            other_set = probes["COPY"][other_tag]
            if det & correct_set:
                labels_by_tag[tag].append("correct")
            elif det & other_set:
                labels_by_tag[tag].append("wrong")
            elif det & family_sseqids_copy:
                labels_by_tag[tag].append("unclassified")
            else:
                labels_by_tag[tag].append("no_signal")
        return labels_by_tag

    labels_b = _classify(base)
    labels_g = _classify(grpo)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    tag_order = ["HIGH", "LOW"]
    run_order = [("base", labels_b), ("GRPO", labels_g)]
    x = np.arange(len(tag_order) * len(run_order))
    tick_labels, groups = [], []
    for t in tag_order:
        for run_label, _ in run_order:
            tick_labels.append(f"{t}\n{run_label}")
            groups.append((t, run_label))

    bottoms = np.zeros(len(x))
    for bucket in buckets_order:
        heights = []
        for (tag, run_label), (_, labels) in zip(groups, run_order * 2):
            # Note: run_order*2 pairs the same run with both tags in order.
            # Rebuild properly:
            pass
        # rebuild heights cleanly
        heights = []
        for (tag, run_label) in groups:
            labels_for_run = labels_b if run_label == "base" else labels_g
            vals = labels_for_run.get(tag, [])
            n = len(vals)
            heights.append(vals.count(bucket) / n if n else 0.0)
        heights = np.array(heights)
        ax.bar(x, heights, bottom=bottoms, color=colors[bucket],
               edgecolor="white", label=bucket, width=0.8)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("fraction of sequences")
    ax.set_ylim(0, 1.0)
    ax.set_title("COPY_ classification — HIGH partially works,\nLOW is ignored in both runs",
                 fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    # Visually separate HIGH from LOW group.
    ax.axvline(len(run_order) - 0.5, color="#d1d5db", linewidth=0.8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"soft_token_copy_buckets.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory (created if missing)")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = load_run(BASE_DIR)
    grpo = load_run(GRPO_DIR)
    prompts = base["gens"]["prompt"].tolist()
    assert prompts == grpo["gens"]["prompt"].tolist(), "prompt order mismatch"

    token_sseqids = load_registry(args.registry)
    probes = resolve_probes(token_sseqids)

    df = compute_discrimination(base, grpo, prompts, probes)
    df.to_csv(args.out / "soft_token_discrimination.csv", index=False)

    plot_forest(df, args.out)
    plot_vec_heatmap(base, grpo, prompts, probes, args.out)
    plot_copy_buckets(base, grpo, prompts, probes, args.out)

    print(f"wrote figures to {args.out}:")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
