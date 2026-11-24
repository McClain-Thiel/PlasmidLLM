#!/usr/bin/env python3
"""Score external baselines and negative controls on the cluster-disjoint panel (n=468).

Runs strictly against `eval/runs/rigor_20261116/inputs/frozen_panel_v3.parquet`
(`panel_main == True`, n=468 held-out sequence clusters):
  1. Verifies local pLannotate 1.2.2 reproduces reference_control features bit-for-bit.
  2. Builds Prompt-Jaccard retrieval picks from cluster_v3/train.parquet and annotates
     unique retrieved plasmids with pLannotate 1.2.2.
  3. Builds GC-matched random and dinucleotide-shuffled controls (seed=42) from the 468
     held-out reference sequences and annotates them with pLannotate 1.2.2.
  4. Scores PlasmidGPT (Shao et al., 2026) and PlasmidGPT-SFT (Cunningham et al., 2025)
     under both 12 kb / 8 kb length caps (ATG/ColE1 equal mean) and full uncapped runs
     across four pairing seeds (42-45), alongside PlasmidLM (supervised base) and
     PlasmidLM (post-GRPO, 3 training seeds x 4 decode seeds).

Outputs `external_baselines_summary.{json,csv}` under `--out-dir`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.neurips_rebuttal.common import (  # noqa: E402
    TOKEN_RE,
    detect,
    is_viable,
)
from eval.scripts.build_negative_controls import (  # noqa: E402
    dinucleotide_shuffle,
    gc_matched_random,
)

PAIRS_TO_SEEDS = [42, 43, 44, 45]
ELIG20_MIN = 20
ELIG10_MIN = 10


def _annotate_one_seq(seq_id: str, seq_str: str) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    from plannotate.annotate import annotate as plannotate_annotate

    seq_str = str(seq_str).upper().strip()
    if len(seq_str) < 100:
        return {
            "seq_id": seq_id,
            "n_features": 0,
            "features": [],
            "coverage_bp": 0,
            "seq_length": len(seq_str),
            "reason": "short_sequence",
        }
    hits = plannotate_annotate(seq_str, is_detailed=True, linear=False)
    features: list[dict[str, Any]] = []
    if hasattr(hits, "iterrows"):
        for _, row in hits.iterrows():
            features.append({
                "sseqid": row.get("sseqid", ""),
                "Feature": row.get("Feature", ""),
                "Type": row.get("Type", ""),
                "pident": float(row.get("pident", 0)),
                "qstart": int(row.get("qstart", 0)),
                "qend": int(row.get("qend", 0)),
                "slen": int(row.get("slen", 0)),
                "length": int(row.get("length", row.get("qend", 0) - row.get("qstart", 0))),
                "percmatch": float(row.get("percmatch", 0)),
                "fragment": bool(row.get("fragment", False)),
                "db": row.get("db", ""),
            })
    return {
        "seq_id": seq_id,
        "n_features": len(features),
        "features": features,
        "coverage_bp": sum(f["length"] for f in features),
        "seq_length": len(seq_str),
    }


def verify_local_plannotate(panel_df: pd.DataFrame, ref_json: Path, n_check: int = 5) -> None:
    ref_list = json.loads(ref_json.read_text())
    ref_by_id = {str(r["seq_id"]): r["features"] for r in ref_list}
    for i in range(n_check):
        seq = str(panel_df.iloc[i]["source_sequence"])
        res = _annotate_one_seq(f"gen_{i}", seq)
        got_sig = sorted((str(f["sseqid"]), round(float(f["percmatch"]), 1), bool(f["fragment"])) for f in res["features"])
        exp_sig = sorted((str(f["sseqid"]), round(float(f["percmatch"]), 1), bool(f["fragment"])) for f in ref_by_id[f"gen_{i}"])
        if got_sig != exp_sig:
            raise AssertionError(
                f"Local pLannotate mismatch on gen_{i}: got {len(got_sig)} vs ref {len(exp_sig)}\n"
                f"  diff got-exp: {set(got_sig) - set(exp_sig)}\n"
                f"  diff exp-got: {set(exp_sig) - set(got_sig)}"
            )
    print(f"Verified local pLannotate 1.2.2 matches reference_control on first {n_check} plasmids.")


def annotate_batch_cached(items: list[tuple[str, str]], cache_path: Path, n_jobs: int) -> dict[str, dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        existing = json.loads(cache_path.read_text())
        by_id = {str(r["seq_id"]): r for r in existing}
        if all(sid in by_id for sid, _ in items):
            print(f"Loaded {len(existing)} cached annotations from {cache_path.name}")
            return by_id
    else:
        by_id = {}

    todo = [(sid, seq) for sid, seq in items if sid not in by_id]
    if todo:
        print(f"Annotating {len(todo)} sequences for {cache_path.name} with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_annotate_one_seq)(sid, seq) for sid, seq in todo
        )
        for r in results:
            by_id[str(r["seq_id"])] = r
        ordered = [by_id[sid] for sid, _ in items]
        tmp_path = cache_path.with_suffix(".tmp.json")
        tmp_path.write_text(json.dumps(ordered, indent=2))
        tmp_path.replace(cache_path)
    return by_id


def compute_panel_metrics(
    requested: list[set[str]],
    feature_lists: list[list[dict[str, Any]]],
    lengths: list[int],
    elig20: list[str],
    elig10: list[str],
) -> dict[str, float]:
    n = len(requested)
    if len(feature_lists) != n or len(lengths) != n:
        raise ValueError("Mismatched panel lengths")
    viable_vals: list[float] = []
    faith_vals: list[float] = []
    f05_vals: list[float] = []
    f10_vals: list[float] = []
    useful_vals: list[float] = []
    prec_vals: list[float] = []
    f1_vals: list[float] = []
    extra_vals: list[float] = []
    extra_oa_vals: list[float] = []
    det_sets: list[set[str]] = []

    for req, feats in zip(requested, feature_lists):
        v = float(is_viable(feats, quality_filter=True))
        det = detect(feats, quality_filter=True)
        det_sets.append(det)
        found = req & det
        faith = len(found) / len(req)
        prec = len(found) / len(det) if det else 0.0
        f1 = (2 * prec * faith / (prec + faith)) if (prec + faith) > 0 else 0.0
        extras = det - req
        extra_oa = sum(1 for t in extras if t.startswith(("<ORI_", "<AMR_")))
        viable_vals.append(v)
        faith_vals.append(faith)
        f05_vals.append(float(faith >= 0.5))
        f10_vals.append(float(faith >= 1.0))
        useful_vals.append(float(bool(v) and faith >= 0.5))
        prec_vals.append(prec)
        f1_vals.append(f1)
        extra_vals.append(float(len(extras)))
        extra_oa_vals.append(float(extra_oa))

    def _mean_lift(tokens: list[str]) -> tuple[float, int]:
        if not tokens:
            return float("nan"), 0
        lifts = []
        gt10 = 0
        for t in tokens:
            pos = [int(t in det) for req, det in zip(requested, det_sets) if t in req]
            neg = [int(t in det) for req, det in zip(requested, det_sets) if t not in req]
            lift = (sum(pos) / len(pos)) - (sum(neg) / len(neg))
            lifts.append(lift)
            if lift > 0.10:
                gt10 += 1
        return float(statistics.mean(lifts)), gt10

    lift20, gt10_20 = _mean_lift(elig20)
    lift10, gt10_10 = _mean_lift(elig10)

    return {
        "n_prompts": float(n),
        "length_bp": float(statistics.mean(lengths)),
        "viable": float(statistics.mean(viable_vals)),
        "faithfulness": float(statistics.mean(faith_vals)),
        "faithful_05": float(statistics.mean(f05_vals)),
        "faithful_10": float(statistics.mean(f10_vals)),
        "useful_05": float(statistics.mean(useful_vals)),
        "precision_all": float(statistics.mean(prec_vals)),
        "f1_hard": float(statistics.mean(f1_vals)),
        "extra_components": float(statistics.mean(extra_vals)),
        "extra_ori_amr": float(statistics.mean(extra_oa_vals)),
        "mean_lift_elig20": lift20,
        "n_lift_elig20_gt10pp": float(gt10_20),
        "mean_lift_elig10": lift10,
        "n_lift_elig10_gt10pp": float(gt10_10),
    }


def _aggregate_metrics(runs: list[dict[str, float]]) -> dict[str, Any]:
    keys = list(runs[0].keys())
    out: dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in runs]
        arr = np.asarray(vals, dtype=float)
        out[k] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return out


def score_unconditional_arm(
    requested: list[set[str]],
    prefix_ann_paths: list[Path],
    elig20: list[str],
    elig10: list[str],
    seeds: list[int] = PAIRS_TO_SEEDS,
) -> dict[str, Any]:
    """Score unconditional baseline averaging prefixes (ATG, ColE1) and pairing seeds."""
    seed_summaries: list[dict[str, float]] = []
    loaded = []
    for p in prefix_ann_paths:
        raw = json.loads(p.read_text())
        by_id = {str(r["seq_id"]): r for r in raw if "error" not in r}
        ids = sorted(by_id)
        loaded.append((by_id, ids))

    for s in seeds:
        prefix_runs: list[dict[str, float]] = []
        for pidx, (by_id, ids) in enumerate(loaded):
            rng = random.Random(s * 1000 + pidx)
            sampled_ids = [ids[rng.randrange(len(ids))] for _ in requested]
            feats = [by_id[sid].get("features", []) for sid in sampled_ids]
            lens = [int(by_id[sid].get("seq_length", 0)) for sid in sampled_ids]
            prefix_runs.append(compute_panel_metrics(requested, feats, lens, elig20, elig10))
        seed_summaries.append({k: float(np.mean([r[k] for r in prefix_runs])) for k in prefix_runs[0]})
    return _aggregate_metrics(seed_summaries)


def score_plasmidlm_arm(
    eval_root: Path,
    model_names: list[str],
    panel_df: pd.DataFrame,
    main_panel: pd.DataFrame,
    requested: list[set[str]],
    elig20: list[str],
    elig10: list[str],
    decode_seeds: list[int] = PAIRS_TO_SEEDS,
) -> dict[str, Any]:
    """Score PlasmidLM models using exact main_panel row mapping from frozen_panel_v3."""
    main_mask = panel_df["panel_main"].to_numpy()
    main_idx = np.flatnonzero(main_mask)
    # Sanity check exact alignment
    assert panel_df.iloc[main_idx]["sample_id"].astype(str).tolist() == main_panel["sample_id"].astype(str).tolist()

    model_seed_summaries: list[dict[str, float]] = []
    for mname in model_names:
        dec_runs: list[dict[str, float]] = []
        for dseed in decode_seeds:
            ann_path = eval_root / "annotations" / mname / f"decode_s{dseed}" / "plannotate_results.json"
            raw = json.loads(ann_path.read_text())
            by_id = {str(r["seq_id"]): r for r in raw}
            feats = [by_id[f"gen_{i}"].get("features", []) for i in main_idx]
            lens = [int(by_id[f"gen_{i}"].get("seq_length", 0)) for i in main_idx]
            dec_runs.append(compute_panel_metrics(requested, feats, lens, elig20, elig10))
        model_seed_summaries.append({k: float(np.mean([r[k] for r in dec_runs])) for k in dec_runs[0]})
    return _aggregate_metrics(model_seed_summaries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=REPO / "eval/runs/rigor_20261116",
    )
    ap.add_argument(
        "--train-parquet",
        type=Path,
        default=REPO / ".paper-audit/rigor_20261116/certification/cluster_v3/train.parquet",
    )
    ap.add_argument("--n-jobs", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--skip-neg-annotate", action="store_true")
    args = ap.parse_args()

    eval_root: Path = args.eval_root
    out_dir = eval_root / "baselines_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_df = pd.read_parquet(eval_root / "inputs/frozen_panel_v3.parquet")
    main_panel = panel_df[panel_df["panel_main"]].reset_index(drop=True)
    requested = [set(TOKEN_RE.findall(str(p))) for p in main_panel["prompt"]]
    n_main = len(requested)
    assert n_main == 468, f"Expected 468 main prompts, got {n_main}"

    counts = collections.Counter(t for r in requested for t in r)
    elig20 = sorted(t for t, c in counts.items() if ELIG20_MIN <= c <= n_main - ELIG20_MIN)
    elig10 = sorted(t for t, c in counts.items() if ELIG10_MIN <= c <= n_main - ELIG10_MIN)
    print(f"Benchmark: n={n_main}, elig>={ELIG20_MIN}: {len(elig20)}, elig>={ELIG10_MIN}: {len(elig10)}")

    # Step 1: Verify local pLannotate reproduces reference_control
    ref_json = eval_root / "annotations/reference_control/plannotate_results.json"
    verify_local_plannotate(panel_df, ref_json, n_check=5)

    # Step 2: Prompt-Jaccard retrieval on cluster_v3 train split
    print(f"Building Jaccard index from {args.train_parquet} ...")
    train_df = pd.read_parquet(
        args.train_parquet, columns=["plasmid_id", "prompt", "sequence", "cluster_id"]
    )
    index: dict[int, set[str]] = collections.defaultdict(set)
    seqs_by_pid: dict[int, str] = {}
    clusters_by_pid: dict[int, str] = {}
    for pid, pr, seq, cid in zip(
        train_df["plasmid_id"], train_df["prompt"], train_df["sequence"], train_df["cluster_id"]
    ):
        ipid = int(pid)
        index[ipid] |= set(TOKEN_RE.findall(str(pr)))
        if ipid not in seqs_by_pid:
            seqs_by_pid[ipid] = str(seq)
            clusters_by_pid[ipid] = str(cid)
    index = {k: v for k, v in index.items() if v}
    postings: dict[str, set[int]] = collections.defaultdict(set)
    for pid, toks in index.items():
        for t in toks:
            postings[t].add(pid)

    picks: list[int] = []
    exact_matches = 0
    for req in requested:
        cands: set[int] = set()
        for t in req:
            cands |= postings.get(t, set())
        if not cands:
            raise ValueError(f"No retrieval candidate for {sorted(req)}")
        best_pid = max(
            cands,
            key=lambda pid: (len(req & index[pid]) / len(req | index[pid]), -pid),
        )
        picks.append(best_pid)
        if req == index[best_pid]:
            exact_matches += 1

    uniq_pids = sorted(set(picks))
    print(f"Retrieval: {exact_matches}/{n_main} exact hard-token matches, {len(uniq_pids)} unique train plasmids")
    (out_dir / "retrieved_plasmid_ids.txt").write_text("\n".join(str(p) for p in picks) + "\n")

    retr_items = [(str(pid), seqs_by_pid[pid]) for pid in uniq_pids]
    retr_by_id = annotate_batch_cached(
        retr_items, out_dir / "retrieval_plannotate_results.json", n_jobs=args.n_jobs
    )
    retr_feats = [retr_by_id[str(pid)].get("features", []) for pid in picks]
    retr_lens = [int(retr_by_id[str(pid)].get("seq_length", len(seqs_by_pid[pid]))) for pid in picks]
    retr_summary = compute_panel_metrics(requested, retr_feats, retr_lens, elig20, elig10)
    retr_summary["exact_token_set_matches"] = float(exact_matches)
    retr_summary["unique_retrieved_plasmids"] = float(len(uniq_pids))
    retr_summary["sequence_novelty_frac"] = 0.0

    # Step 3: Negative controls (Random GC-matched & Dinucleotide-shuffled on n=468 reference sequences)
    rng = np.random.default_rng(42)
    rand_items: list[tuple[str, str]] = []
    shuf_items: list[tuple[str, str]] = []
    for i, row in main_panel.iterrows():
        seq = str(row["source_sequence"]).upper().strip()
        gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
        rand_items.append((f"rand_{i}", gc_matched_random(len(seq), gc, rng)))
        shuf_items.append((f"shuf_{i}", dinucleotide_shuffle(seq, rng)))

    if not args.skip_neg_annotate:
        rand_by_id = annotate_batch_cached(
            rand_items, out_dir / "random_plannotate_results.json", n_jobs=args.n_jobs
        )
        shuf_by_id = annotate_batch_cached(
            shuf_items, out_dir / "shuffled_plannotate_results.json", n_jobs=args.n_jobs
        )
        rand_summary = compute_panel_metrics(
            requested,
            [rand_by_id[f"rand_{i}"].get("features", []) for i in range(n_main)],
            [int(rand_by_id[f"rand_{i}"].get("seq_length", len(rand_items[i][1]))) for i in range(n_main)],
            elig20, elig10,
        )
        shuf_summary = compute_panel_metrics(
            requested,
            [shuf_by_id[f"shuf_{i}"].get("features", []) for i in range(n_main)],
            [int(shuf_by_id[f"shuf_{i}"].get("seq_length", len(shuf_items[i][1]))) for i in range(n_main)],
            elig20, elig10,
        )
    else:
        rand_summary = shuf_summary = {}

    # Step 4: Score PlasmidLM models & external unconditional PlasmidGPT baselines on n=468
    raw_dir = out_dir / "raw"
    lc_dir = raw_dir / "length_controlled"

    results: dict[str, dict[str, Any]] = {}
    results["PlasmidLM (post-GRPO)"] = score_plasmidlm_arm(
        eval_root, ["exp2_v2_s42", "exp2_v2_s43", "exp2_v2_s44"],
        panel_df, main_panel, requested, elig20, elig10,
    )
    results["PlasmidLM (supervised base)"] = score_plasmidlm_arm(
        eval_root, ["cluster_base_s42"],
        panel_df, main_panel, requested, elig20, elig10,
    )
    results["Prompt-Jaccard retrieval"] = retr_summary
    results["PlasmidGPT (12 kb cap)"] = score_unconditional_arm(
        requested,
        [
            lc_dir / "plasmidgpt_shao_atg/length_panels/cap_12000bp/plannotate_results.json",
            lc_dir / "plasmidgpt_shao_cole1/length_panels/cap_12000bp/plannotate_results.json",
        ],
        elig20, elig10,
    )
    results["PlasmidGPT-SFT (12 kb cap)"] = score_unconditional_arm(
        requested,
        [
            lc_dir / "plasmidgpt_cunningham_atg/length_panels/cap_12000bp/plannotate_results.json",
            lc_dir / "plasmidgpt_cunningham_cole1/length_panels/cap_12000bp/plannotate_results.json",
        ],
        elig20, elig10,
    )
    results["PlasmidGPT (8 kb cap)"] = score_unconditional_arm(
        requested,
        [
            lc_dir / "plasmidgpt_shao_atg/length_panels/cap_08000bp/plannotate_results.json",
            lc_dir / "plasmidgpt_shao_cole1/length_panels/cap_08000bp/plannotate_results.json",
        ],
        elig20, elig10,
    )
    results["PlasmidGPT-SFT (8 kb cap)"] = score_unconditional_arm(
        requested,
        [
            lc_dir / "plasmidgpt_cunningham_atg/length_panels/cap_08000bp/plannotate_results.json",
            lc_dir / "plasmidgpt_cunningham_cole1/length_panels/cap_08000bp/plannotate_results.json",
        ],
        elig20, elig10,
    )
    results["PlasmidGPT (full uncapped)"] = score_unconditional_arm(
        requested, [raw_dir / "comp_shao_plannotate_results.json"], elig20, elig10,
    )
    results["PlasmidGPT-SFT (full uncapped)"] = score_unconditional_arm(
        requested, [raw_dir / "comp_cunn_plannotate_results.json"], elig20, elig10,
    )
    if rand_summary:
        results["Random GC-matched"] = rand_summary
        results["Dinucleotide-shuffled"] = shuf_summary

    # Real held-out reference control
    main_idx = np.flatnonzero(panel_df["panel_main"].to_numpy())
    ref_raw = json.loads(ref_json.read_text())
    ref_by_id = {str(r["seq_id"]): r for r in ref_raw}
    results["Real Addgene (held-out)"] = compute_panel_metrics(
        requested,
        [ref_by_id[f"gen_{i}"].get("features", []) for i in main_idx],
        [int(ref_by_id[f"gen_{i}"].get("seq_length", 0)) for i in main_idx],
        elig20, elig10,
    )

    payload = {
        "n_prompts": n_main,
        "elig20_tokens": elig20,
        "elig10_tokens": elig10,
        "results": results,
    }
    (out_dir / "external_baselines_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "external_baselines_summary.csv", index=False)

    print("\n=== Unified n=468 Cluster-Disjoint Benchmark Comparison ===")
    hdr = f"{'Model':<32}{'Len(kb)':>8}{'Viable':>9}{'F>=0.5':>9}{'Useful':>9}{'Prec':>8}{'Lift20':>9}{'Lift10':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in results.items():
        print(
            f"{name:<32}{m['length_bp']/1000:>8.1f}{m['viable']*100:>8.1f}%"
            f"{m['faithful_05']*100:>8.1f}%{m['useful_05']*100:>8.1f}%"
            f"{m['precision_all']*100:>7.1f}%"
            f"{m['mean_lift_elig20']*100:>+8.1f}%{m['mean_lift_elig10']*100:>+8.1f}%"
        )


if __name__ == "__main__":
    main()
