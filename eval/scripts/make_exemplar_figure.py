#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow>=14.0",
#     "numpy>=1.26",
#     "matplotlib>=3.7",
#     "dna_features_viewer>=3.1",
# ]
# ///
"""Exemplar gallery — three rows of (prompt, generated plasmid map).

Pickslices three concrete prompts from `runs/grpo_plannotate_full_20260408/`:
  • gen_271 — clean success, 5/5 requested tokens recovered, simple kanamycin/EGFP
  • gen_223 — typical single-shot, 3/6 recovered, common mismatches
  • gen_7   — COPY_LOW failure: model emits a high-copy origin (ColE1, f1) when
              the prompt requested a low-copy origin (p15A)

Each row shows:
  - Left: the prompt rendered as a tidy component list (✓ for found, ✗ missing).
  - Right: a linear pLannotate map of the generated sequence, colored by type
           (origin / selection / promoter / CDS / regulatory / other), with the
           requested-token components labelled.

Reads from a flat bucket cache (default `.paper-audit/bucket/`).

Usage:
    uv run --script eval/scripts/make_exemplar_figure.py \\
        --cache .paper-audit/bucket --out eval/figures/paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dna_features_viewer import GraphicFeature, GraphicRecord


GRPO_RUN = "grpo_plannotate_full_20260408"
EXEMPLARS = [
    {"seq_id": "gen_271", "label": "(a) clean success",
     "tag":   r"5/5 requested tokens recovered ($f_i = 1.00$)"},
    {"seq_id": "gen_223", "label": "(b) typical single-shot",
     "tag":   r"3/6 requested tokens recovered ($f_i = 0.50$)"},
    {"seq_id": "gen_7",   "label": "(c) COPY_LOW failure",
     "tag":   "high-copy origins (ColE1, f1) emitted when COPY_LOW was requested"},
]

# Type → color, scoped to plannotate Types we expect to encounter
TYPE_COLOR = {
    "rep_origin":     "#3b82f6",  # blue (origins)
    "CDS":            "#a855f7",  # purple (coding)
    "promoter":       "#10b981",  # green (promoters)
    "terminator":     "#f59e0b",  # amber (terminators)
    "polyA_signal":   "#f59e0b",
    "ncRNA":          "#ec4899",  # pink
    "repeat_region":  "#94a3b8",  # slate
    "mobile_element": "#7c3aed",  # violet
    "misc_feature":   "#cbd5e1",  # light slate
    "5'UTR":          "#fde68a",
    "3'UTR":          "#fde68a",
}
DEFAULT_COLOR = "#cbd5e1"

# Token category prefixes → display name / color used in the prompt panel.
TOKEN_CAT_COLOR = {
    "AMR":      "#ef4444",
    "ORI":      "#3b82f6",
    "PROM":     "#10b981",
    "ELEM":     "#94a3b8",
    "REPORTER": "#a855f7",
    "TAG":      "#ec4899",
    "VEC":      "#f59e0b",
    "SP":       "#f59e0b",
    "COPY":     "#f59e0b",
    "BB":       "#f59e0b",
    "REP":      "#7c3aed",
}


def load_data(cache: Path):
    fid = pd.read_parquet(cache / GRPO_RUN / "metrics/prompt_fidelity_per_seq.parquet")
    plan = json.load(open(cache / GRPO_RUN / "plannotate_results.json"))
    plan = {p["seq_id"]: p for p in plan}
    return fid, plan


def parse_prompt_tokens(prompt: str) -> list[str]:
    """Pull <NAME> tokens from the prompt string, in order, dropping <BOS>/<SEQ>."""
    import re
    toks = re.findall(r"<([^>]+)>", prompt)
    return [t for t in toks if t not in ("BOS", "SEQ")]


def features_for_record(plan_entry: dict) -> list[GraphicFeature]:
    """Convert pLannotate features into dna_features_viewer GraphicFeatures."""
    feats: list[GraphicFeature] = []
    for f in plan_entry.get("features", []):
        start = int(f.get("qstart", 1)) - 1
        end = int(f.get("qend", start + 1))
        if end <= start:
            continue
        ftype = f.get("Type", "")
        color = TYPE_COLOR.get(ftype, DEFAULT_COLOR)
        # Use the human-readable feature name; trim long ones.
        name = str(f.get("Feature", ""))
        if len(name) > 18:
            name = name[:16] + "…"
        feats.append(GraphicFeature(
            start=start, end=end, strand=+1,
            color=color, label=name, thickness=14,
            fontdict={"fontsize": 7},
        ))
    return feats


def render_prompt_panel(ax, prompt_tokens: list[str], found_tokens: set[str]) -> None:
    """Render the prompt as a vertical list with ✓/✗ indicators."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # We split tokens into "verifiable" (hard tokens used in fidelity) and
    # the rest (soft metadata: VEC_, SP_, COPY_, BB_).
    HARD_PREFIXES = {"AMR", "ORI", "PROM", "ELEM", "REPORTER", "TAG", "REP"}
    hard = [t for t in prompt_tokens if t.split("_", 1)[0] in HARD_PREFIXES]
    soft = [t for t in prompt_tokens if t.split("_", 1)[0] not in HARD_PREFIXES]

    rows = []
    if soft:
        rows.append(("metadata", None, None))
        for t in soft:
            cat = t.split("_", 1)[0]
            rows.append((t, TOKEN_CAT_COLOR.get(cat, "#94a3b8"), "soft"))
    if hard:
        rows.append(("requested components", None, None))
        for t in hard:
            cat = t.split("_", 1)[0]
            ok = t in found_tokens
            rows.append((t, TOKEN_CAT_COLOR.get(cat, "#94a3b8"), "ok" if ok else "miss"))

    n = len(rows)
    y_step = 1.0 / max(n + 1, 1)
    for i, (label, color, kind) in enumerate(rows):
        y = 1.0 - (i + 0.5) * y_step
        if kind is None:
            ax.text(0.02, y, label.upper(), fontsize=8, color="#6b7280",
                    fontweight="bold")
            continue
        marker = "✓" if kind == "ok" else ("·" if kind == "soft" else "✗")
        marker_color = "#059669" if kind == "ok" else ("#94a3b8" if kind == "soft"
                                                        else "#dc2626")
        ax.text(0.04, y, marker, fontsize=11, color=marker_color, fontweight="bold")
        ax.text(0.12, y, label, fontsize=9, color="#1f2937",
                fontfamily="monospace")
        # Tiny color swatch on the right
        ax.add_patch(plt.Rectangle((0.95, y - 0.012), 0.04, 0.022,
                                    transform=ax.transAxes,
                                    facecolor=color, edgecolor="none",
                                    clip_on=False))


def make_figure(cache: Path, out_dir: Path) -> None:
    fid, plan = load_data(cache)

    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.05, 0.04, 3.0],
                           hspace=0.55, wspace=0.05)

    for row, ex in enumerate(EXEMPLARS):
        sid = ex["seq_id"]
        if sid not in plan:
            print(f"  warning: {sid} not in plannotate results, skipping row")
            continue
        meta = fid[fid["seq_id"] == sid].iloc[0]
        prompt_toks = parse_prompt_tokens(meta["prompt"])
        found = set(meta["found_tokens"])

        # left panel: prompt
        ax_left = fig.add_subplot(gs[row, 0])
        render_prompt_panel(ax_left, prompt_toks, found)
        ax_left.text(0.0, 1.13, ex["label"], transform=ax_left.transAxes,
                      fontsize=10, fontweight="bold", color="#1f2937")
        ax_left.text(0.0, 1.04, ex["tag"], transform=ax_left.transAxes,
                      fontsize=8, color="#4b5563")

        # right panel: plasmid map
        ax_right = fig.add_subplot(gs[row, 2])
        seq_len = plan[sid]["seq_length"]
        feats = features_for_record(plan[sid])
        record = GraphicRecord(
            sequence_length=seq_len,
            features=feats,
        )
        record.plot(ax=ax_right, with_ruler=True, figure_width=8)
        ax_right.set_title(
            f"{sid}  ·  {seq_len:,} bp  ·  {plan[sid]['n_features']} detected features",
            loc="left", fontsize=9, color="#374151", pad=2,
        )

    # Single shared legend at the bottom for type → color
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["rep_origin"],     label="origin"),
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["CDS"],            label="CDS"),
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["promoter"],       label="promoter"),
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["polyA_signal"],   label="terminator / polyA"),
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["ncRNA"],          label="ncRNA"),
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["repeat_region"],  label="repeat / mobile"),
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR["misc_feature"],   label="misc"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exemplar_gallery.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "exemplar_gallery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  exemplar_gallery.pdf — 3 rows: {[e['seq_id'] for e in EXEMPLARS]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("eval/figures/paper"))
    args = p.parse_args()
    if not args.cache.is_dir():
        raise SystemExit(f"cache dir does not exist: {args.cache}")
    make_figure(args.cache, args.out)


if __name__ == "__main__":
    main()
