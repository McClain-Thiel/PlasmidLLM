#!/usr/bin/env python3
"""Reproduce auxiliary manuscript diagnostics, sensitivity checks, and tables.

Covers all auxiliary numbers reported alongside ``summarize_rigor_evaluation.py``
on the ``frozen_panel_v3`` sequence-cluster-disjoint archive:
  1. Clean-prompt sensitivity masking noisy <ELEM_IRES>/<ELEM_TRACRRNA> labels (Table 6).
  2. Complete BoK-2 / BoK-4 metrics including orthogonal AMRFinderPlus recall (Tables 1 & 5).
  3. Per-token resistance marker 3-seed means +- SDs across all 11 AMR classes (Table 7).
  4. Termination stratification (<EOS> vs. 4,096-token cap; Table 9 Panel A).
  5. Compositional generalization on seen (n=360) vs. unseen (n=108) specs (Table 9 Panel B).
  6. NCBI AMRFinderPlus --plus virulence screen across all 7,488 generations (Section 6).
  7. Exhaustive circular megablast sequence novelty vs. all training plasmids (Table 9 Panel C).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.neurips_rebuttal.common import TOKEN_RE  # noqa: E402
from experiments.neurips_rebuttal.summarize_rigor_evaluation import (  # noqa: E402
    _attach_covered_amr_recall,
    _best_of_k_frame,
    _paired_summary_with_p,
)
from pretraining.build_cluster_split import canonical_circle  # noqa: E402
from pretraining.verify_cluster_split import circular_intervals, interval_union  # noqa: E402

NOISY_PROMPT_TOKENS = frozenset({"<ELEM_IRES>", "<ELEM_TRACRRNA>"})
DECODE_SEEDS = [42, 43, 44, 45]
POST_GRPO_MODELS = ("exp2_v2_s42", "exp2_v2_s43", "exp2_v2_s44")


def _default_split_dir(repo_root: Path, rigor_dir: Path) -> Path:
    candidates = [
        rigor_dir / "split" / "cluster_v3",
        repo_root / "data" / "cluster_v3",
        repo_root / ".paper-audit" / "rigor_20261116" / "certification" / "cluster_v3",
    ]
    for cand in candidates:
        if (cand / "train.parquet").exists():
            return cand
    return candidates[0]


def _seed_mean_std(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_seed": [float(v) for v in arr],
    }


def _arm_seed_summary(
    base_df: pd.DataFrame,
    grpo_df: pd.DataFrame,
    panel: pd.DataFrame,
    metrics: list[str],
    *,
    decode_seeds: list[int] | None = None,
    iters: int = 10000,
) -> dict[str, Any]:
    dseeds = list(decode_seeds or DECODE_SEEDS)
    id_panel = panel[["sample_id", "cluster_id", "prompt"]]
    out: dict[str, Any] = {}
    for metric in metrics:
        res = _paired_summary_with_p(
            base_df,
            grpo_df,
            id_panel,
            metric=metric,
            decode_seeds=dseeds,
            fixed_baseline=True,
            iters=iters,
        )
        per_seed = [
            float(
                grpo_df[grpo_df["training_seed"] == ts]
                .groupby("sample_id")[metric]
                .mean()
                .mean()
            )
            for ts in sorted(grpo_df["training_seed"].unique())
        ]
        res["treatment_std"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
        res["per_seed"] = per_seed
        out[metric] = res
    return out


def compute_clean_prompt_sensitivity(
    main_panel: pd.DataFrame,
    ref_main: pd.DataFrame,
    base_main: pd.DataFrame,
    grpo_main: pd.DataFrame,
    *,
    iters: int = 10000,
) -> dict[str, Any]:
    """Mask <ELEM_IRES> and <ELEM_TRACRRNA> from requested tokens and rescore (Table 6)."""
    clean_req_map: dict[str, set[str]] = {}
    for row in main_panel.itertuples():
        req = set(TOKEN_RE.findall(str(row.prompt))) - NOISY_PROMPT_TOKENS
        if not req:
            raise ValueError(f"Empty clean prompt for sample_id={row.sample_id}")
        clean_req_map[str(row.sample_id)] = req

    def _add_clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        faith_c, f1_c, u05_c, u075_c, u10_c = [], [], [], [], []
        for row in out.itertuples():
            req = clean_req_map[str(row.sample_id)]
            det = set(json.loads(row.detected_tokens_json))
            found = req & det
            rec = len(found) / len(req)
            prec = len(found) / len(det) if det else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            via = float(row.viable) > 0.5
            faith_c.append(rec)
            f1_c.append(f1)
            u05_c.append(float(via and rec >= 0.5))
            u075_c.append(float(via and rec >= 0.75))
            u10_c.append(float(via and rec >= 1.0 - 1e-9))
        out["faithfulness_clean"] = faith_c
        out["f1_clean"] = f1_c
        out["useful_05_clean"] = u05_c
        out["useful_075_clean"] = u075_c
        out["useful_10_clean"] = u10_c
        return out

    ref_c = _add_clean_cols(ref_main)
    base_c = _add_clean_cols(base_main)
    grpo_c = _add_clean_cols(grpo_main)

    metrics = ["faithfulness_clean", "useful_05_clean", "useful_075_clean", "useful_10_clean", "f1_clean"]
    return {
        "n_prompts": int(len(clean_req_map)),
        "excluded_tokens": sorted(NOISY_PROMPT_TOKENS),
        "reference_addgene": {m: float(ref_c[m].mean()) for m in metrics},
        "paired": _arm_seed_summary(base_c, grpo_c, main_panel, metrics, iters=iters),
    }


def compute_complete_best_of_k(
    main_panel: pd.DataFrame,
    base_main: pd.DataFrame,
    grpo_main: pd.DataFrame,
    covered_tokens: set[str],
    *,
    iters: int = 10000,
) -> dict[str, Any]:
    """Compute BoK-2 and BoK-4 across all Table 1 and Table 5 endpoints incl. AMRFinderPlus."""
    base_cov = _attach_covered_amr_recall(base_main, covered_tokens)
    grpo_cov = _attach_covered_amr_recall(grpo_main, covered_tokens)

    all_amr_ids = set(base_cov.loc[base_cov["has_req_amr"], "sample_id"].astype(str))
    cov_amr_ids = set(base_cov.loc[base_cov["has_req_covered_amr"], "sample_id"].astype(str))
    amr_panel = main_panel[main_panel["sample_id"].astype(str).isin(all_amr_ids)]
    cov_panel = main_panel[main_panel["sample_id"].astype(str).isin(cov_amr_ids)]

    base_metrics = [
        "useful_05",
        "viable",
        "faithfulness",
        "f1_hard",
        "precision_all",
        "extra_components",
        "extra_ori_amr",
    ]
    out: dict[str, Any] = {}
    for k in (2, 4):
        b_sel = _best_of_k_frame(base_cov, DECODE_SEEDS, k).assign(seed=0)
        g_sel = _best_of_k_frame(grpo_cov, DECODE_SEEDS, k).assign(seed=0)
        entry = _arm_seed_summary(
            b_sel, g_sel, main_panel, base_metrics, decode_seeds=[0], iters=iters
        )
        entry["amr_orthogonal_intact_recall"] = _arm_seed_summary(
            b_sel[b_sel["sample_id"].astype(str).isin(all_amr_ids)],
            g_sel[g_sel["sample_id"].astype(str).isin(all_amr_ids)],
            amr_panel,
            ["amr_orthogonal_intact_recall"],
            decode_seeds=[0],
            iters=iters,
        )["amr_orthogonal_intact_recall"]
        entry["amr_orthogonal_intact_covered_recall"] = _arm_seed_summary(
            b_sel[b_sel["sample_id"].astype(str).isin(cov_amr_ids)],
            g_sel[g_sel["sample_id"].astype(str).isin(cov_amr_ids)],
            cov_panel,
            ["amr_orthogonal_intact_covered_recall"],
            decode_seeds=[0],
            iters=iters,
        )["amr_orthogonal_intact_covered_recall"]
        out[f"K={k}"] = entry
    return out


def compute_amr_per_token_breakdown(
    ref_main: pd.DataFrame,
    base_main: pd.DataFrame,
    grpo_main: pd.DataFrame,
) -> dict[str, Any]:
    """Compute per-token pLannotate and orthogonal AMRFinderPlus 3-seed SDs (Table 7)."""

    def _rates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        counts: dict[str, list[int]] = {}
        for row in df.itertuples():
            req = json.loads(row.requested_amr_json)
            det_plan = set(json.loads(row.detected_tokens_json))
            det_orth = set(json.loads(row.orthogonal_intact_detected_json))
            for tok in req:
                bucket = counts.setdefault(tok, [0, 0, 0])
                bucket[0] += 1
                bucket[1] += int(tok in det_plan)
                bucket[2] += int(tok in det_orth)
        return {
            tok: {
                "requested": c[0],
                "plannotate": c[1] / c[0],
                "amrfinder_intact": c[2] / c[0],
            }
            for tok, c in sorted(counts.items())
        }

    ref_rates = _rates(ref_main)
    base_rates = _rates(base_main)
    seed_rates = {
        s: _rates(grpo_main[grpo_main["training_seed"] == s])
        for s in sorted(grpo_main["training_seed"].unique())
    }

    rows: dict[str, Any] = {}
    for tok, ref_info in ref_rates.items():
        plan_seeds = [seed_rates[s][tok]["plannotate"] for s in sorted(seed_rates)]
        orth_seeds = [seed_rates[s][tok]["amrfinder_intact"] for s in sorted(seed_rates)]
        rows[tok] = {
            "n_req": int(ref_info["requested"]),
            "addgene": {
                "plannotate": ref_info["plannotate"],
                "amrfinder_intact": ref_info["amrfinder_intact"],
            },
            "base": {
                "plannotate": base_rates[tok]["plannotate"],
                "amrfinder_intact": base_rates[tok]["amrfinder_intact"],
            },
            "post_grpo": {
                "plannotate": _seed_mean_std(plan_seeds),
                "amrfinder_intact": _seed_mean_std(orth_seeds),
            },
        }
    return rows


def compute_termination_stratification(
    base_main: pd.DataFrame,
    grpo_main: pd.DataFrame,
    covered_tokens: set[str],
) -> dict[str, Any]:
    """Stratify single-shot outcomes by clean <EOS> vs. 4,096-token cap (Table 9 Panel A)."""
    base_cov = _attach_covered_amr_recall(base_main, covered_tokens)
    grpo_cov = _attach_covered_amr_recall(grpo_main, covered_tokens)

    def _summarize_sub(df: pd.DataFrame, n_total: int) -> dict[str, Any]:
        return {
            "n_draws": int(len(df)),
            "fraction": float(len(df) / n_total),
            "mean_length_bp": float(df["length"].mean()),
            "useful_05": float(df["useful_05"].mean()),
            "viable": float(df["viable"].mean()),
            "faithfulness": float(df["faithfulness"].mean()),
            "f1_hard": float(df["f1_hard"].mean()),
            "precision_all": float(df["precision_all"].mean()),
            "extra_components": float(df["extra_components"].mean()),
            "amr_cov9_recall": float(df["amr_orthogonal_intact_covered_recall"].dropna().mean()),
        }

    out: dict[str, Any] = {}
    for arm_name, df in (("base", base_cov), ("post_grpo", grpo_cov)):
        is_trunc = df["truncated"].astype(float) > 0.5
        clean = df[~is_trunc]
        cap = df[is_trunc]
        mid = clean[(clean["length"] >= 4000) & (clean["length"] <= 10000)]

        entry: dict[str, Any] = {
            "n_total": int(len(df)),
            "clean_eos": _summarize_sub(clean, len(df)),
            "cap_4096": _summarize_sub(cap, len(df)),
            "clean_eos_4_to_10kb": _summarize_sub(mid, len(df)),
        }
        if df["training_seed"].nunique() > 1:
            clean_fracs = [
                float((df.loc[df["training_seed"] == s, "truncated"] <= 0.5).mean())
                for s in sorted(df["training_seed"].unique())
            ]
            entry["clean_eos"]["fraction_seed_mean_std"] = _seed_mean_std(clean_fracs)
        out[arm_name] = entry
    return out


def compute_seen_unseen_generalization(
    main_panel: pd.DataFrame,
    base_main: pd.DataFrame,
    grpo_main: pd.DataFrame,
    split_dir: Path,
    covered_tokens: set[str],
    *,
    iters: int = 10000,
) -> dict[str, Any]:
    """Disaggregate seen (n=360) vs. unseen (n=108) hard-token combos (Table 9 Panel B)."""
    train_df = pd.read_parquet(split_dir / "train.parquet", columns=["prompt"])
    train_hard = {frozenset(TOKEN_RE.findall(str(p))) for p in train_df["prompt"]}

    main = main_panel.copy()
    main["is_seen"] = [frozenset(TOKEN_RE.findall(str(p))) in train_hard for p in main["prompt"]]

    base_cov = _attach_covered_amr_recall(base_main, covered_tokens)
    grpo_cov = _attach_covered_amr_recall(grpo_main, covered_tokens)
    main_metrics = ["useful_05", "viable", "faithfulness", "f1_hard"]

    out: dict[str, Any] = {}
    for label, flag in (("seen_in_train", True), ("unseen_novel", False)):
        sub_panel = main[main["is_seen"] == flag]
        sub_ids = set(sub_panel["sample_id"].astype(str))
        b_sub = base_cov[base_cov["sample_id"].astype(str).isin(sub_ids)]
        g_sub = grpo_cov[grpo_cov["sample_id"].astype(str).isin(sub_ids)]
        res = _arm_seed_summary(b_sub, g_sub, sub_panel, main_metrics, iters=iters)

        cov_ids = set(b_sub.loc[b_sub["has_req_covered_amr"], "sample_id"].astype(str))
        cov_panel = sub_panel[sub_panel["sample_id"].astype(str).isin(cov_ids)]
        res["amr_orthogonal_intact_covered_recall"] = _arm_seed_summary(
            b_sub[b_sub["sample_id"].astype(str).isin(cov_ids)],
            g_sub[g_sub["sample_id"].astype(str).isin(cov_ids)],
            cov_panel,
            ["amr_orthogonal_intact_covered_recall"],
            iters=iters,
        )["amr_orthogonal_intact_covered_recall"]
        res["n_clusters"] = int(len(sub_panel))
        res["n_cov_amr_clusters"] = int(len(cov_panel))
        out[label] = res
    return out


def compute_virulence_screen(
    annot_dir: Path,
    main_seq_ids_by_run: dict[tuple[str, int], set[str]] | None = None,
) -> dict[str, Any]:
    """Audit archived AMRFinderPlus --plus outputs for non-AMR virulence hits (Section 6)."""
    targets: list[tuple[str, int, Path]] = [
        ("reference_control", 42, annot_dir / "reference_control" / "amrfinder.tsv"),
    ]
    for mname in ("cluster_base_s42", *POST_GRPO_MODELS):
        for dseed in DECODE_SEEDS:
            targets.append((mname, dseed, annot_dir / mname / f"decode_s{dseed}" / "amrfinder.tsv"))

    virulence_hits: list[dict[str, str]] = []
    all_rows_by_type: dict[str, int] = {}
    main_virulence = 0
    for mname, dseed, tsv_path in targets:
        with tsv_path.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                etype = str(row.get("Type", "")).strip().upper()
                all_rows_by_type[etype] = all_rows_by_type.get(etype, 0) + 1
                if etype == "VIRULENCE":
                    pid = row.get("Protein id") or row.get("Protein identifier") or ""
                    sid = pid.rsplit("_", 1)[0]
                    virulence_hits.append({"model": mname, "seed": str(dseed), "protein": pid})
                    if main_seq_ids_by_run is None or sid in main_seq_ids_by_run.get(
                        (mname, dseed), set()
                    ):
                        main_virulence += 1

    return {
        "n_evaluated_panels": len(targets),
        "n_generation_panels": len(targets) - 1,
        "n_main_generations_screened": 4 * 4 * 468,
        "virulence_hits_total": len(virulence_hits),
        "virulence_hits_main_panel": main_virulence,
        "row_counts_by_type": all_rows_by_type,
        "virulence_examples": virulence_hits[:10],
    }


def _blast_shard(args: tuple[list[tuple[str, str]], Path, int]):
    q_records, db_pref, n_targets = args
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as qf:
        qpath = Path(qf.name)
        q_lens: dict[str, int] = {}
        for qid, seq in q_records:
            q_lens[qid] = len(seq)
            if len(seq) >= 28:
                qf.write(f">{qid}\n{seq + seq}\n")
    try:
        cmd = [
            "blastn",
            "-task",
            "megablast",
            "-query",
            str(qpath),
            "-db",
            str(db_pref),
            "-outfmt",
            "6 qseqid sseqid pident qstart qend sstart send qlen slen",
            "-perc_identity",
            "90",
            "-max_target_seqs",
            str(n_targets),
            "-dust",
            "no",
            "-evalue",
            "1e-10",
            "-num_threads",
            "1",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1024 * 1024
        )
        best_bidir = {qid: 0.0 for qid in q_lens}
        best_qcov = {qid: 0.0 for qid in q_lens}
        union_q: dict[str, list[tuple[int, int]]] = {qid: [] for qid in q_lens}
        cur = None
        qi: list[tuple[int, int]] = []
        si: list[tuple[int, int]] = []
        ql0 = sl0 = 0

        def _finish() -> None:
            if cur is None:
                return
            q, _ = cur
            qc = interval_union(qi) / ql0
            sc = interval_union(si) / sl0
            bd = min(qc, sc)
            if bd > best_bidir[q]:
                best_bidir[q] = bd
            if qc > best_qcov[q]:
                best_qcov[q] = qc

        assert proc.stdout is not None
        for line in proc.stdout:
            q, s, ident, qs, qe, ss, se, ql, sl = line.rstrip().split("\t")
            key = (q, s)
            if key != cur:
                _finish()
                cur, qi, si = key, [], []
            ql0, sl0 = int(ql) // 2, int(sl)
            if float(ident) >= 90.0:
                c_qi = circular_intervals(int(qs), int(qe), ql0)
                qi.extend(c_qi)
                union_q[q].extend(c_qi)
                si.extend(circular_intervals(int(ss), int(se), sl0))
        _finish()
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read() if proc.stderr else "blastn error"
            raise RuntimeError(f"blastn failed ({rc}): {err}")
        mosaic = {
            q: (interval_union(union_q[q]) / q_lens[q] if q_lens[q] > 0 else 0.0)
            for q in q_lens
        }
        return best_bidir, best_qcov, mosaic
    finally:
        qpath.unlink(missing_ok=True)


def run_exhaustive_megablast_novelty(
    rigor_dir: Path,
    split_dir: Path,
    main_panel: pd.DataFrame,
    obs_df: pd.DataFrame,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Run exhaustive circular megablast (-max_target_seqs |train|) at decode seed 42."""
    main_ids = set(main_panel["sample_id"].astype(str))
    m_obs = obs_df[obs_df["sample_id"].astype(str).isin(main_ids)]
    train_df = pd.read_parquet(
        split_dir / "train.parquet", columns=["sequence_id", "sequence"]
    ).drop_duplicates("sequence_id")
    n_train = len(train_df)

    def _dist_stats(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float) * 100.0
        return {
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "mean": float(np.mean(arr)),
        }

    with tempfile.TemporaryDirectory(prefix="exhaust_nov_") as tmp:
        work = Path(tmp)
        tfa = work / "train.fa"
        with tfa.open("w") as f:
            for r in train_df.itertuples():
                f.write(f">{r.sequence_id}\n{canonical_circle(r.sequence)}\n")
        dbp = work / "train_db"
        subprocess.run(
            ["makeblastdb", "-in", str(tfa), "-dbtype", "nucl", "-out", str(dbp)],
            check=True,
            capture_output=True,
        )

        all_q: list[tuple[str, str]] = [
            (f"ref:{idx}", str(r["source_sequence"]).upper())
            for idx, r in main_panel.reset_index(drop=True).iterrows()
        ]
        arms = ("cluster_base_s42", *POST_GRPO_MODELS)
        for mname in arms:
            gparq = pd.read_parquet(
                rigor_dir / "evaluations_v1" / mname / "decode_s42" / "generations.parquet"
            ).reset_index(drop=True)
            valid = set(
                m_obs[(m_obs["model_name"] == mname) & (m_obs["seed"] == 42)]["seq_id"]
            )
            for idx, r in gparq.iterrows():
                sid = str(r["seq_id"]) if "seq_id" in gparq.columns else f"gen_{idx}"
                if sid in valid:
                    all_q.append((f"{mname}:{sid}", str(r["sequence"]).upper()))

        shards = [all_q[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_blast_shard, [(sh, dbp, n_train) for sh in shards]))

        b_bidir, b_qcov, b_mos = {}, {}, {}
        for d1, d2, d3 in outs:
            b_bidir.update(d1)
            b_qcov.update(d2)
            b_mos.update(d3)

    summary: dict[str, Any] = {"n_train_unique": n_train, "per_model": {}}
    for tag in ("ref", *arms):
        keys = [k for k in b_bidir if k.startswith(tag + ":")]
        nov_vals = [1.0 if b_bidir[k] < 0.90 else 0.0 for k in keys]
        summary["per_model"][tag] = {
            "n": len(keys),
            "sequence_novelty_lt90_bidir_pct": float(np.mean(nov_vals) * 100.0),
            "best_bidir_cov_pct": _dist_stats([b_bidir[k] for k in keys]),
            "best_qcov_pct": _dist_stats([b_qcov[k] for k in keys]),
            "mosaic_alltrain_qcov_pct": _dist_stats([b_mos[k] for k in keys]),
        }

    grpo_agg: dict[str, Any] = {
        "sequence_novelty_lt90_bidir_pct": _seed_mean_std(
            [summary["per_model"][s]["sequence_novelty_lt90_bidir_pct"] for s in POST_GRPO_MODELS]
        )
    }
    for sub_key in ("best_bidir_cov_pct", "best_qcov_pct", "mosaic_alltrain_qcov_pct"):
        meds = [summary["per_model"][s][sub_key]["median"] for s in POST_GRPO_MODELS]
        means = [summary["per_model"][s][sub_key]["mean"] for s in POST_GRPO_MODELS]
        grpo_agg[sub_key] = {
            "median_mean": float(np.mean(meds)),
            "median_std": float(np.std(meds, ddof=1)),
            "per_seed_medians": meds,
            "mean_of_means": float(np.mean(means)),
            "mean_std": float(np.std(means, ddof=1)),
        }
    summary["post_grpo_summary"] = grpo_agg
    return summary


def run_manuscript_diagnostics(
    *,
    rigor_dir: Path,
    split_dir: Path,
    out_dir: Path | None = None,
    novelty_json: Path | None = None,
    run_blast: bool = False,
    iters: int = 10000,
    workers: int = 8,
) -> dict[str, Any]:
    """Compute all auxiliary manuscript tables and save manuscript_diagnostics.json."""
    results_dir = rigor_dir / "results_v1"
    panel_df = pd.read_parquet(rigor_dir / "inputs" / "frozen_panel_v3.parquet")
    obs_df = pd.read_parquet(results_dir / "all_observations.parquet")
    summary = json.loads((results_dir / "summary.json").read_text())
    covered_tokens = set(summary["reference_control"]["amrfinder_covered_tokens"])

    main_panel = panel_df[panel_df["panel_main"].astype(bool)].copy()
    main_ids = set(main_panel["sample_id"].astype(str))
    main_obs = obs_df[obs_df["sample_id"].astype(str).isin(main_ids)].copy()

    ref_main = main_obs[main_obs["model_name"] == "reference_control"].copy()
    base_main = main_obs[main_obs["model_family"] == "cluster_base_s42"].copy()
    grpo_main = main_obs[main_obs["model_family"] == "exp2_v2"].copy()

    payload: dict[str, Any] = {
        "clean_prompt_sensitivity": compute_clean_prompt_sensitivity(
            main_panel, ref_main, base_main, grpo_main, iters=iters
        ),
        "best_of_k_complete": compute_complete_best_of_k(
            main_panel, base_main, grpo_main, covered_tokens, iters=iters
        ),
        "amr_per_token": compute_amr_per_token_breakdown(ref_main, base_main, grpo_main),
        "termination_stratification": compute_termination_stratification(
            base_main, grpo_main, covered_tokens
        ),
        "seen_vs_unseen_specs": compute_seen_unseen_generalization(
            main_panel, base_main, grpo_main, split_dir, covered_tokens, iters=iters
        ),
        "virulence_screen": compute_virulence_screen(rigor_dir / "annotations"),
    }

    if run_blast:
        nov = run_exhaustive_megablast_novelty(
            rigor_dir, split_dir, main_panel, obs_df, workers=workers
        )
        if out_dir is not None:
            (out_dir / "sequence_novelty_summary.json").write_text(json.dumps(nov, indent=2) + "\n")
        payload["sequence_novelty"] = nov
    else:
        nov_path = novelty_json or (results_dir / "sequence_novelty_summary.json")
        if nov_path.exists():
            payload["sequence_novelty"] = json.loads(nov_path.read_text())

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manuscript_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_rigor = REPO / "eval" / "runs" / "rigor_20261116"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigor-dir", type=Path, default=default_rigor)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--novelty-json", type=Path, default=None)
    parser.add_argument("--run-blast", action="store_true")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    split_dir = args.split_dir or _default_split_dir(REPO, args.rigor_dir)
    out_dir = args.out_dir or (args.rigor_dir / "results_v1")
    payload = run_manuscript_diagnostics(
        rigor_dir=args.rigor_dir,
        split_dir=split_dir,
        out_dir=out_dir,
        novelty_json=args.novelty_json,
        run_blast=args.run_blast,
        iters=args.iters,
        workers=args.workers,
    )
    vir = payload["virulence_screen"]
    clean = payload["clean_prompt_sensitivity"]["paired"]["useful_10_clean"]
    seen_u = payload["seen_vs_unseen_specs"]["seen_in_train"]["useful_05"]
    unseen_u = payload["seen_vs_unseen_specs"]["unseen_novel"]["useful_05"]
    print(
        f"Saved manuscript_diagnostics.json to {out_dir} | "
        f"clean T=1.00: {clean['baseline_mean']:.1%} -> {clean['treatment_mean']:.1%} | "
        f"seen={seen_u['treatment_mean']:.1%}, unseen={unseen_u['treatment_mean']:.1%} | "
        f"virulence_hits={vir['virulence_hits_total']}"
    )


if __name__ == "__main__":
    main()
