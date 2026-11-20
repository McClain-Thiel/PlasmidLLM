#!/usr/bin/env python3
"""Aggregate per-panel annotations and compute paired three-seed rigor statistics.

Consumes verified ``per_sequence_metrics.parquet`` outputs across the 11 evaluated
models (4 decoding seeds each) plus the ``reference_control`` source sequence panel,
stratifying strictly by ``panel_main`` and each of the four minority panels without
merging panels or dropping annotation failures.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.neurips_rebuttal.common import (  # noqa: E402
    CATEGORIES,
    TOKEN_RE,
    category_of,
    detect,
    is_viable,
    load_annotations,
    summarize_paired_training_seeds,
    wilson,
)
from experiments.neurips_rebuttal.orthogonal import (  # noqa: E402
    AMR_TOKEN_RE,
    complete_protein_ids,
)
from post_training.scorers.plannotate_v2 import TAG_PEPTIDES  # noqa: E402

LOW_COPY_ORIS = frozenset({"<ORI_P15A>", "<ORI_PSC101>"})
HIGH_COPY_ORIS = frozenset({"<ORI_COLE1>", "<ORI_RSF>"})
UNCLASSIFIED_ORIS = frozenset({"<ORI_F1>", "<ORI_SV40>", "<ORI_2MU>"})
MINORITY_PANEL_ORDER = (
    "<ORI_P15A>",
    "<ORI_PSC101>",
    "<AMR_CHLORAMPHENICOL>",
    "<COPY_LOW>",
)
TAG_TOKEN_RE = re.compile(r"<TAG_[A-Z0-9_]+>")
PRIMARY_HOLM_METRICS = (
    "useful_05",
    "viable",
    "faithfulness",
    "extra_ori_amr",
    "amr_orthogonal_intact_recall",
    "tag_recall_prodigal_peptide",
)
MAIN_METRICS = (
    "useful_05",
    "useful_10",
    "viable",
    "faithfulness",
    "faithful_05",
    "faithful_10",
    "f1_hard",
    "precision_all",
    "extra_components",
    "has_extra_component",
    "extra_ori_amr",
    "has_unrequested_ori",
    "has_unrequested_amr",
    "length",
    "gc",
    "has_eos",
    "truncated",
    "lt_1kb",
    "low_complexity_frac",
)
MODEL_FAMILY_SPECS = {
    "hf_base": {"models": ["hf_base"], "seeds": [42]},
    "cluster_base_s42": {"models": ["cluster_base_s42"], "seeds": [42]},
    "exp1_v1": {"models": ["exp1_v1_s42", "exp1_v1_s43", "exp1_v1_s44"], "seeds": [42, 43, 44]},
    "exp1_v2": {"models": ["exp1_v2_s42", "exp1_v2_s43", "exp1_v2_s44"], "seeds": [42, 43, 44]},
    "exp2_v2": {"models": ["exp2_v2_s42", "exp2_v2_s43", "exp2_v2_s44"], "seeds": [42, 43, 44]},
}


def parse_dustmasker_intervals(intervals_path: Path, lengths: dict[str, int]) -> dict[str, float]:
    """Compute per-sequence low-complexity fraction from ``dustmasker -outfmt interval``."""
    raw_intervals: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    current_id: str | None = None
    for raw_line in intervals_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_id = line[1:].split()[0]
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", line)
        if match and current_id is not None:
            start, end = int(match.group(1)), int(match.group(2))
            if current_id not in lengths:
                raise ValueError(f"Unknown sequence ID in dustmasker intervals: {current_id}")
            seq_len = lengths[current_id]
            lo = max(0, min(start, seq_len))
            hi = max(lo, min(end + 1, seq_len))
            if hi > lo:
                raw_intervals[current_id].append((lo, hi))

    fractions: dict[str, float] = {}
    for seq_id, seq_len in lengths.items():
        if seq_len <= 0:
            fractions[seq_id] = 0.0
            continue
        spans = sorted(raw_intervals.get(seq_id, []))
        merged_bp = 0
        cur_lo, cur_hi = -1, -1
        for lo, hi in spans:
            if lo > cur_hi:
                merged_bp += cur_hi - cur_lo if cur_hi > cur_lo else 0
                cur_lo, cur_hi = lo, hi
            elif hi > cur_hi:
                cur_hi = hi
        if cur_hi > cur_lo:
            merged_bp += cur_hi - cur_lo
        fractions[seq_id] = float(merged_bp / seq_len)
    return fractions


def detect_prodigal_peptide_tags(proteins_faa: Path) -> dict[str, set[str]]:
    """Detect ``TAG_PEPTIDES`` motifs inside complete, unambiguous circular Prodigal ORFs."""
    from Bio import SeqIO

    complete = complete_protein_ids(proteins_faa)
    detected: dict[str, set[str]] = collections.defaultdict(set)
    for record in SeqIO.parse(str(proteins_faa), "fasta"):
        if record.id not in complete:
            continue
        seq_id = record.id.rsplit("_", 1)[0]
        peptide = str(record.seq).removesuffix("*")
        for tag, motifs in TAG_PEPTIDES.items():
            if any(motif in peptide for motif in motifs):
                detected[seq_id].add(tag)
    return dict(detected)


def classify_copy_low(detected_tokens: set[str]) -> dict[str, float]:
    """Compute mutually exclusive bacterial origin-family proxies for COPY_LOW analysis."""
    has_low = bool(detected_tokens & LOW_COPY_ORIS)
    has_high = bool(detected_tokens & HIGH_COPY_ORIS)
    has_unclassified = bool(detected_tokens & UNCLASSIFIED_ORIS)
    has_any_ori = any(token.startswith("<ORI_") for token in detected_tokens)

    buckets = {
        "copy_low_compatible_only": float(has_low and not has_high),
        "copy_low_conflicting_high_and_low": float(has_low and has_high),
        "copy_low_high_only": float(has_high and not has_low),
        "copy_low_unclassified_only": float(not has_low and not has_high and has_unclassified),
        "copy_low_no_origin": float(not has_any_ori),
    }
    total = sum(buckets.values())
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(
            f"COPY_LOW origin buckets do not sum to 1.0 ({total}) for {sorted(detected_tokens)}"
        )
    buckets["copy_low_any_low"] = float(has_low)
    buckets["multi_bacterial_origin"] = float(len(detected_tokens & (LOW_COPY_ORIS | HIGH_COPY_ORIS)) >= 2)
    return buckets


def _parse_minority_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return [str(x) for x in parsed]
    if value is None:
        return []
    return [str(x) for x in list(value)]


def _orthogonal_map(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        sid = str(row["sequence_id"])
        mapping[sid] = {
            "detected": {str(x) for x in list(row["detected"])},
            "recall": float(row["recall"]),
        }
    return mapping


def score_single_panel_dataframe(
    *,
    panel_df: pd.DataFrame,
    generations_df: pd.DataFrame,
    plannotate_annotations: dict[str, list[dict]],
    orthogonal_all_df: pd.DataFrame,
    orthogonal_intact_df: pd.DataFrame,
    peptide_tags: dict[str, set[str]],
    low_complexity: dict[str, float],
    model_name: str,
    model_family: str,
    training_seed: int,
    decode_seed: int,
) -> pd.DataFrame:
    """Build per-sequence evaluation records for one (model, decode_seed) or reference panel."""
    if len(generations_df) != len(panel_df):
        raise ValueError(
            f"Row count mismatch for {model_name} seed={decode_seed}: "
            f"{len(generations_df)} != {len(panel_df)}"
        )
    gen_reset = generations_df.reset_index(drop=True)
    panel_reset = panel_df.reset_index(drop=True)
    for col in ("sample_id", "cluster_id", "prompt"):
        if col in gen_reset.columns and not gen_reset[col].astype(str).equals(panel_reset[col].astype(str)):
            raise ValueError(f"Column {col} in {model_name} seed={decode_seed} does not match panel")

    orth_all_map = _orthogonal_map(orthogonal_all_df)
    orth_intact_map = _orthogonal_map(orthogonal_intact_df)

    rows: list[dict[str, Any]] = []
    for idx, panel_row in panel_reset.iterrows():
        seq_id = f"gen_{idx}"
        if seq_id not in plannotate_annotations:
            raise ValueError(f"Missing pLannotate output for {seq_id} in {model_name}")
        if seq_id not in low_complexity:
            raise ValueError(f"Missing dustmasker output for {seq_id} in {model_name}")

        prompt = str(panel_row["prompt"])
        req = set(TOKEN_RE.findall(prompt))
        if not req:
            raise ValueError(f"Prompt {seq_id} lacks hard tokens")
        feats = plannotate_annotations[seq_id]
        det = detect(feats, quality_filter=True)
        viable = is_viable(feats, quality_filter=True)
        found = req & det
        faith = len(found) / len(req)
        prec_cond = len(found) / len(det) if det else math.nan
        prec_all = len(found) / len(det) if det else 0.0
        f1_hard = (2 * prec_all * faith / (prec_all + faith)) if (prec_all + faith) > 0 else 0.0
        extras = det - req
        extra_ori = sum(1 for tok in extras if tok.startswith("<ORI_"))
        extra_amr = sum(1 for tok in extras if tok.startswith("<AMR_"))

        req_amr = set(AMR_TOKEN_RE.findall(prompt))
        has_req_amr = bool(req_amr)
        if has_req_amr:
            if seq_id not in orth_all_map or seq_id not in orth_intact_map:
                raise ValueError(f"Missing AMRFinder orthogonal output for {seq_id} in {model_name}")
            amr_plan_rec = len(req_amr & det) / len(req_amr)
            orth_all_entry = orth_all_map[seq_id]
            orth_intact_entry = orth_intact_map[seq_id]
            amr_orth_all_rec = orth_all_entry["recall"]
            amr_orth_intact_rec = orth_intact_entry["recall"]
            orth_all_det = sorted(orth_all_entry["detected"])
            orth_intact_det = sorted(orth_intact_entry["detected"])
        else:
            amr_plan_rec = math.nan
            amr_orth_all_rec = math.nan
            amr_orth_intact_rec = math.nan
            orth_all_det = []
            orth_intact_det = []

        req_tags = set(TAG_TOKEN_RE.findall(prompt))
        has_req_tag = bool(req_tags)
        pep_det = peptide_tags.get(seq_id, set())
        if has_req_tag:
            tag_plan_rec = len(req_tags & det) / len(req_tags)
            pep_hits = {
                tag for tag in req_tags
                if (tag in pep_det) or (tag == "<TAG_GST>" and "<TAG_GST>" in det)
            }
            tag_pep_rec = len(pep_hits) / len(req_tags)
        else:
            tag_plan_rec = math.nan
            tag_pep_rec = math.nan

        gen_row = gen_reset.iloc[idx]
        length = int(gen_row["length"]) if "length" in gen_row else len(str(gen_row["sequence"]))
        seq_str = str(gen_row.get("sequence", ""))
        gc = (
            float(gen_row["gc"])
            if "gc" in gen_row
            else (seq_str.upper().count("G") + seq_str.upper().count("C")) / max(len(seq_str), 1)
        )
        has_eos = float(bool(gen_row.get("has_eos", True)))
        truncated = float(bool(gen_row.get("truncated", False)))
        copy_proxies = classify_copy_low(det)

        rows.append({
            "sample_id": str(panel_row["sample_id"]),
            "cluster_id": str(panel_row["cluster_id"]),
            "prompt": prompt,
            "panel_main": bool(panel_row["panel_main"]),
            "minority_panels_json": json.dumps(_parse_minority_list(panel_row.get("minority_panels"))),
            "model_name": model_name,
            "model_family": model_family,
            "training_seed": int(training_seed),
            "seed": int(decode_seed),
            "seq_id": seq_id,
            "viable": float(viable),
            "faithfulness": float(faith),
            "faithful_05": float(faith >= 0.5),
            "faithful_10": float(faith >= 1.0),
            "useful_05": float(viable and faith >= 0.5),
            "useful_10": float(viable and faith >= 1.0),
            "precision_cond": prec_cond,
            "precision_all": float(prec_all),
            "f1_hard": float(f1_hard),
            "extra_components": float(len(extras)),
            "has_extra_component": float(bool(extras)),
            "extra_ori_amr": float(extra_ori + extra_amr),
            "has_unrequested_ori": float(extra_ori > 0),
            "has_unrequested_amr": float(extra_amr > 0),
            "n_requested_hard": int(len(req)),
            "n_found": int(len(found)),
            "requested_tokens_json": json.dumps(sorted(req)),
            "detected_tokens_json": json.dumps(sorted(det)),
            "length": float(length),
            "gc": float(gc),
            "has_eos": has_eos,
            "truncated": truncated,
            "lt_1kb": float(length < 1000),
            "low_complexity_frac": float(low_complexity[seq_id]),
            "has_req_amr": has_req_amr,
            "amr_plannotate_recall": amr_plan_rec,
            "amr_orthogonal_all_recall": amr_orth_all_rec,
            "amr_orthogonal_intact_recall": amr_orth_intact_rec,
            "requested_amr_json": json.dumps(sorted(req_amr)),
            "orthogonal_all_detected_json": json.dumps(orth_all_det),
            "orthogonal_intact_detected_json": json.dumps(orth_intact_det),
            "has_req_tag": has_req_tag,
            "tag_recall_plannotate": tag_plan_rec,
            "tag_recall_prodigal_peptide": tag_pep_rec,
            "requested_tag_json": json.dumps(sorted(req_tags)),
            "prodigal_peptide_detected_json": json.dumps(sorted(pep_det)),
            **copy_proxies,
            "hit_ori_p15a": float("<ORI_P15A>" in det),
            "hit_ori_psc101": float("<ORI_PSC101>" in det),
            "hit_amr_chloramphenicol_plannotate": float("<AMR_CHLORAMPHENICOL>" in det),
            "hit_amr_chloramphenicol_intact_amrfinder": float("<AMR_CHLORAMPHENICOL>" in orth_intact_det),
        })
    return pd.DataFrame(rows)


def _cluster_seed_bootstrap_p(
    baseline_mat: np.ndarray,
    treatment_mat: np.ndarray,
    cluster_ids: list[str],
    iters: int = 10000,
    seed: int = 42,
) -> float:
    clusters = sorted(set(cluster_ids))
    masks = [np.array([c == cluster for c in cluster_ids]) for cluster in clusters]
    sums = np.stack([(treatment_mat - baseline_mat)[:, mask].sum(axis=1) for mask in masks], axis=1)
    counts = np.array([mask.sum() for mask in masks])
    rng = np.random.default_rng(seed)
    samples = np.empty(iters)
    for i in range(iters):
        seed_idx = rng.integers(baseline_mat.shape[0], size=baseline_mat.shape[0])
        cluster_idx = rng.integers(len(clusters), size=len(clusters))
        samples[i] = sums[np.ix_(seed_idx, cluster_idx)].sum() / (
            len(seed_idx) * counts[cluster_idx].sum()
        )
    p_le = (np.sum(samples <= 0.0) + 1) / (iters + 1)
    p_ge = (np.sum(samples >= 0.0) + 1) / (iters + 1)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        scaled = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = float(running_max)
    return adjusted


def _paired_summary_with_p(
    baseline_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    panel: pd.DataFrame,
    metric: str,
    decode_seeds: list[int],
    fixed_baseline: bool,
    iters: int,
) -> dict[str, Any]:
    summary = summarize_paired_training_seeds(
        baseline=baseline_df,
        treatment=treatment_df,
        panel=panel,
        metric=metric,
        decode_seeds=decode_seeds,
        fixed_baseline=fixed_baseline,
        iters=iters,
    )
    identity = ["sample_id", "cluster_id", "prompt"]
    reference = panel[identity].set_index("sample_id").sort_index()

    def _matrix(df: pd.DataFrame) -> np.ndarray:
        seeds = sorted(df.training_seed.unique().tolist())
        rows_out = []
        for tseed in seeds:
            decodes = []
            for dseed in decode_seeds:
                sub = df[(df.training_seed == tseed) & (df.seed == dseed)].set_index("sample_id").sort_index()
                decodes.append(sub[metric].to_numpy(dtype=float))
            rows_out.append(np.mean(decodes, axis=0))
        return np.asarray(rows_out)

    b_mat = _matrix(baseline_df)
    t_mat = _matrix(treatment_df)
    if fixed_baseline:
        b_mat = np.repeat(b_mat, t_mat.shape[0], axis=0)
    summary["p_value"] = _cluster_seed_bootstrap_p(
        b_mat, t_mat, reference.cluster_id.tolist(), iters=iters
    )
    return summary


def _compute_reference_control_summary(
    panel_df: pd.DataFrame, ref_df: pd.DataFrame
) -> tuple[dict[str, Any], set[str], pd.DataFrame]:
    """Audit reference_control across hard tokens, TAGs, and AMRFinderPlus classes."""
    main_ids = set(panel_df[panel_df.panel_main]["sample_id"].astype(str))
    main_ref = ref_df[ref_df.sample_id.isin(main_ids)]

    amr_counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"requested": 0, "plannotate_found": 0, "amrfinder_all_found": 0, "amrfinder_intact_found": 0}
    )
    for row in main_ref.itertuples():
        det_plan = set(json.loads(row.detected_tokens_json))
        det_all = set(json.loads(row.orthogonal_all_detected_json))
        det_intact = set(json.loads(row.orthogonal_intact_detected_json))
        for tok in json.loads(row.requested_amr_json):
            amr_counts[tok]["requested"] += 1
            amr_counts[tok]["plannotate_found"] += int(tok in det_plan)
            amr_counts[tok]["amrfinder_all_found"] += int(tok in det_all)
            amr_counts[tok]["amrfinder_intact_found"] += int(tok in det_intact)

    cov_rows = []
    covered_tokens: set[str] = set()
    for tok, counts in sorted(amr_counts.items()):
        req_n = counts["requested"]
        intact_n = counts["amrfinder_intact_found"]
        if intact_n > 0:
            covered_tokens.add(tok)
        cov_rows.append({
            "token": tok,
            "requested": req_n,
            "plannotate_found": counts["plannotate_found"],
            "plannotate_rate": counts["plannotate_found"] / max(req_n, 1),
            "amrfinder_all_found": counts["amrfinder_all_found"],
            "amrfinder_all_rate": counts["amrfinder_all_found"] / max(req_n, 1),
            "amrfinder_intact_found": intact_n,
            "amrfinder_intact_rate": intact_n / max(req_n, 1),
            "amrfinder_covered_on_reference": intact_n > 0,
        })
    cov_df = pd.DataFrame(cov_rows)

    def _safe_mean(series: pd.Series) -> float | None:
        vals = series.dropna()
        return float(vals.mean()) if not vals.empty else None

    summary = {
        "n_main": int(len(main_ref)),
        "viable": _safe_mean(main_ref["viable"]),
        "faithfulness": _safe_mean(main_ref["faithfulness"]),
        "faithful_05": _safe_mean(main_ref["faithful_05"]),
        "faithful_10": _safe_mean(main_ref["faithful_10"]),
        "useful_05": _safe_mean(main_ref["useful_05"]),
        "precision_all": _safe_mean(main_ref["precision_all"]),
        "amr_plannotate_recall": _safe_mean(main_ref.loc[main_ref.has_req_amr, "amr_plannotate_recall"]),
        "amr_orthogonal_all_recall": _safe_mean(main_ref.loc[main_ref.has_req_amr, "amr_orthogonal_all_recall"]),
        "amr_orthogonal_intact_recall": _safe_mean(main_ref.loc[main_ref.has_req_amr, "amr_orthogonal_intact_recall"]),
        "tag_recall_plannotate": _safe_mean(main_ref.loc[main_ref.has_req_tag, "tag_recall_plannotate"]),
        "tag_recall_prodigal_peptide": _safe_mean(main_ref.loc[main_ref.has_req_tag, "tag_recall_prodigal_peptide"]),
        "amrfinder_covered_tokens": sorted(covered_tokens),
    }
    return summary, covered_tokens, cov_df


def _attach_covered_amr_recall(df: pd.DataFrame, covered_tokens: set[str]) -> pd.DataFrame:
    out = df.copy()
    has_cov = []
    cov_rec = []
    for req_json, det_json in zip(out["requested_amr_json"], out["orthogonal_intact_detected_json"]):
        req_cov = set(json.loads(req_json)) & covered_tokens
        if not req_cov:
            has_cov.append(False)
            cov_rec.append(math.nan)
        else:
            det = set(json.loads(det_json))
            has_cov.append(True)
            cov_rec.append(len(req_cov & det) / len(req_cov))
    out["has_req_covered_amr"] = has_cov
    out["amr_orthogonal_intact_covered_recall"] = cov_rec
    return out


def _best_of_k_frame(family_df: pd.DataFrame, decode_seeds: list[int], k: int) -> pd.DataFrame:
    chosen_seeds = set(decode_seeds[:k])
    sub = family_df[family_df.seed.isin(chosen_seeds)].copy()
    sub = sub.sort_values(
        ["training_seed", "sample_id", "n_found", "viable", "faithfulness", "seed"],
        ascending=[True, True, False, False, False, True],
    )
    return sub.drop_duplicates(["training_seed", "sample_id"], keep="first").reset_index(drop=True)


def _summarize_family_stratum(
    family_df: pd.DataFrame,
    stratum_panel: pd.DataFrame,
    decode_seeds: list[int],
    metrics: list[str],
) -> dict[str, Any]:
    """Summarize one model family on one stratum (averaging decodes per seed, then seeds)."""
    valid_ids = set(stratum_panel["sample_id"].astype(str))
    sub = family_df[family_df.sample_id.isin(valid_ids)]
    seed_means: dict[str, list[float]] = collections.defaultdict(list)
    for tseed, seed_grp in sub.groupby("training_seed"):
        per_prompt = seed_grp.groupby("sample_id", as_index=False).mean(numeric_only=True)
        for metric in metrics:
            if metric not in per_prompt.columns:
                continue
            vals = per_prompt[metric].dropna()
            if not vals.empty:
                seed_means[metric].append(float(vals.mean()))
        prec_cond_vals = seed_grp["precision_cond"].dropna()
        if not prec_cond_vals.empty:
            seed_means["precision_cond"].append(float(prec_cond_vals.mean()))

    summary: dict[str, Any] = {
        "n_prompts": int(len(stratum_panel)),
        "n_clusters": int(stratum_panel["cluster_id"].nunique()),
        "n_training_seeds": int(sub["training_seed"].nunique()),
        "decode_seeds": decode_seeds,
        "metrics": {},
    }
    for metric, vals in seed_means.items():
        arr = np.asarray(vals, dtype=float)
        summary["metrics"][metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "per_seed": [float(v) for v in arr],
        }
    if "useful_05" in sub.columns:
        k_int = int(round(sub["useful_05"].sum()))
        n_int = int(len(sub))
        lo, hi = wilson(k_int, n_int)
        summary["useful_05_pooled_wilson_pct"] = [lo, hi]
    return summary


def _build_category_tag_rows(
    combined_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    main_panel: pd.DataFrame,
) -> list[dict[str, Any]]:
    main_ids = set(main_panel["sample_id"].astype(str))
    rows_out: list[dict[str, Any]] = []
    sources: list[tuple[str, pd.DataFrame]] = [("reference_control", ref_df)]
    for fam in MODEL_FAMILY_SPECS:
        sources.append((fam, combined_df[combined_df.model_family == fam]))

    for fam_name, df in sources:
        sub = df[df.sample_id.isin(main_ids)]
        cat_req: collections.Counter = collections.Counter()
        cat_hit: collections.Counter = collections.Counter()
        tag_req: collections.Counter = collections.Counter()
        tag_plan_hit: collections.Counter = collections.Counter()
        tag_pep_hit: collections.Counter = collections.Counter()
        for row in sub.itertuples():
            req = json.loads(row.requested_tokens_json)
            det = set(json.loads(row.detected_tokens_json))
            pep_det = set(json.loads(row.prodigal_peptide_detected_json))
            for tok in req:
                cat = category_of(tok)
                cat_req[cat] += 1
                cat_hit[cat] += int(tok in det)
                if cat == "TAG":
                    tag_req[tok] += 1
                    tag_plan_hit[tok] += int(tok in det)
                    tag_pep_hit[tok] += int((tok in pep_det) or (tok == "<TAG_GST>" and tok in det))
        for cat in CATEGORIES:
            req_n = cat_req[cat]
            rows_out.append({
                "model_family": fam_name,
                "kind": "category",
                "name": cat,
                "requested": int(req_n),
                "plannotate_hit": int(cat_hit[cat]),
                "plannotate_rate": cat_hit[cat] / max(req_n, 1),
                "peptide_hit": np.nan,
                "peptide_rate": np.nan,
            })
        for tok in sorted(tag_req):
            req_n = tag_req[tok]
            rows_out.append({
                "model_family": fam_name,
                "kind": "tag_token",
                "name": tok,
                "requested": int(req_n),
                "plannotate_hit": int(tag_plan_hit[tok]),
                "plannotate_rate": tag_plan_hit[tok] / max(req_n, 1),
                "peptide_hit": int(tag_pep_hit[tok]),
                "peptide_rate": tag_pep_hit[tok] / max(req_n, 1),
            })
    return rows_out


def summarize_rigor_evaluation(
    panel_df: pd.DataFrame,
    panel_manifest: dict[str, Any],
    ref_df: pd.DataFrame,
    model_frames: list[pd.DataFrame],
    decode_seeds: list[int] | None = None,
    iters: int = 10000,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run stratified summaries, Best-of-K, paired cluster+seed tests, and optional export."""
    decode_seeds = list(decode_seeds or [42, 43, 44, 45])
    if not model_frames:
        raise ValueError("No evaluated model frames provided")
    combined = pd.concat(model_frames, ignore_index=True)

    ref_summary, covered_tokens, ref_amr_cov_df = _compute_reference_control_summary(panel_df, ref_df)
    ref_df = _attach_covered_amr_recall(ref_df, covered_tokens)
    combined = _attach_covered_amr_recall(combined, covered_tokens)

    strata: dict[str, pd.DataFrame] = {
        "main": panel_df[panel_df.panel_main].copy(),
    }
    minority_lists = panel_df["minority_panels"].map(_parse_minority_list)
    for tok in MINORITY_PANEL_ORDER:
        slug = "minority_" + tok.strip("<>").lower()
        strata[slug] = panel_df[minority_lists.map(lambda labels, t=tok: t in labels)].copy()

    all_metrics = list(MAIN_METRICS) + [
        "amr_plannotate_recall",
        "amr_orthogonal_all_recall",
        "amr_orthogonal_intact_recall",
        "amr_orthogonal_intact_covered_recall",
        "tag_recall_plannotate",
        "tag_recall_prodigal_peptide",
        "copy_low_compatible_only",
        "copy_low_conflicting_high_and_low",
        "copy_low_any_low",
        "copy_low_high_only",
        "copy_low_unclassified_only",
        "copy_low_no_origin",
        "multi_bacterial_origin",
        "hit_ori_p15a",
        "hit_ori_psc101",
        "hit_amr_chloramphenicol_plannotate",
        "hit_amr_chloramphenicol_intact_amrfinder",
    ]

    model_summaries: dict[str, dict[str, Any]] = {}
    for fam in MODEL_FAMILY_SPECS:
        fam_df = combined[combined.model_family == fam]
        if fam_df.empty:
            continue
        for stratum_name, stratum_panel in strata.items():
            model_summaries.setdefault(stratum_name, {})[fam] = _summarize_family_stratum(
                fam_df, stratum_panel, decode_seeds, all_metrics
            )
    for stratum_name in strata:
        model_summaries.setdefault(stratum_name, {})

    # Best-of-K (K in {1, 2, 4}) on the main stratum + paired deltas.
    main_panel = strata["main"]
    main_ids = set(main_panel["sample_id"].astype(str))
    bok_summary: dict[str, dict[str, Any]] = {}
    bok_frames: dict[tuple[str, int], pd.DataFrame] = {}
    for k in (1, 2, 4):
        bok_summary[f"K={k}"] = {}
        for fam in MODEL_FAMILY_SPECS:
            fam_main = combined[(combined.model_family == fam) & (combined.sample_id.isin(main_ids))]
            if fam_main.empty:
                continue
            sel = _best_of_k_frame(fam_main, decode_seeds, k)
            bok_frames[(fam, k)] = sel
            per_seed = sel.groupby("training_seed", as_index=False).mean(numeric_only=True)
            bok_summary[f"K={k}"][fam] = {
                metric: {
                    "mean": float(per_seed[metric].mean()),
                    "std": float(per_seed[metric].std(ddof=1)) if len(per_seed) > 1 else 0.0,
                }
                for metric in ("viable", "faithfulness", "faithful_05", "faithful_10", "useful_05", "extra_ori_amr")
            }

    bok_paired: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for comp_name, base_fam, treat_fam, fixed_base in (
        ("exp1_v2_vs_exp1_v1", "exp1_v1", "exp1_v2", False),
        ("exp2_v2_vs_cluster_base_s42", "cluster_base_s42", "exp2_v2", True),
        ("exp1_v2_vs_hf_base", "hf_base", "exp1_v2", True),
    ):
        for k in (1, 2, 4):
            b_sel = bok_frames.get((base_fam, k))
            t_sel = bok_frames.get((treat_fam, k))
            if b_sel is None or t_sel is None or b_sel.empty or t_sel.empty:
                continue
            b_single = b_sel.assign(seed=0)
            t_single = t_sel.assign(seed=0)
            for metric in ("useful_05", "viable", "faithfulness", "faithful_05", "faithful_10"):
                res = _paired_summary_with_p(
                    b_single, t_single, main_panel[["sample_id", "cluster_id", "prompt"]],
                    metric=metric, decode_seeds=[0], fixed_baseline=fixed_base, iters=iters,
                )
                bok_paired[f"{comp_name}:K={k}"][metric] = res

    # Paired comparisons across strata and metric sets.
    comparisons_spec = [
        ("exp1_v2_vs_exp1_v1", "exp1_v1", "exp1_v2", False, False, "Primary 3-seed GRPO-v2 vs GRPO-v1"),
        ("exp1_v1_vs_hf_base", "hf_base", "exp1_v1", True, False, "Exp1 GRPO-v1 vs pinned HF base"),
        ("exp1_v2_vs_hf_base", "hf_base", "exp1_v2", True, False, "Exp1 GRPO-v2 vs pinned HF base"),
        ("exp2_v2_vs_cluster_base_s42", "cluster_base_s42", "exp2_v2", True, False, "Secondary Exp2 GRPO-v2 vs fixed cluster-disjoint base"),
        ("exp2_v2_vs_exp1_v2", "exp1_v2", "exp2_v2", False, True, "Observational cross-base comparison (1 pretrained base per arm)"),
        ("cluster_base_s42_vs_hf_base", "hf_base", "cluster_base_s42", True, True, "Observational base comparison (1 pretraining seed per arm; cluster CI only)"),
    ]

    stratum_metrics = {
        "main": [
            ("useful_05", None),
            ("useful_10", None),
            ("viable", None),
            ("faithfulness", None),
            ("faithful_05", None),
            ("faithful_10", None),
            ("f1_hard", None),
            ("precision_all", None),
            ("extra_components", None),
            ("extra_ori_amr", None),
            ("has_unrequested_ori", None),
            ("has_unrequested_amr", None),
            ("amr_plannotate_recall", "has_req_amr"),
            ("amr_orthogonal_all_recall", "has_req_amr"),
            ("amr_orthogonal_intact_recall", "has_req_amr"),
            ("amr_orthogonal_intact_covered_recall", "has_req_covered_amr"),
            ("tag_recall_plannotate", "has_req_tag"),
            ("tag_recall_prodigal_peptide", "has_req_tag"),
        ],
        "minority_ori_p15a": [("hit_ori_p15a", None), ("useful_05", None), ("viable", None), ("faithfulness", None), ("extra_ori_amr", None)],
        "minority_ori_psc101": [("hit_ori_psc101", None), ("useful_05", None), ("viable", None), ("faithfulness", None), ("extra_ori_amr", None)],
        "minority_amr_chloramphenicol": [
            ("hit_amr_chloramphenicol_plannotate", None),
            ("hit_amr_chloramphenicol_intact_amrfinder", None),
            ("useful_05", None),
            ("viable", None),
            ("faithfulness", None),
            ("extra_ori_amr", None),
        ],
        "minority_copy_low": [
            ("copy_low_compatible_only", None),
            ("copy_low_conflicting_high_and_low", None),
            ("copy_low_any_low", None),
            ("copy_low_high_only", None),
            ("copy_low_unclassified_only", None),
            ("copy_low_no_origin", None),
            ("multi_bacterial_origin", None),
            ("useful_05", None),
            ("viable", None),
            ("faithfulness", None),
        ],
    }

    paired_rows: list[dict[str, Any]] = []
    paired_dict: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)

    for comp_name, base_fam, treat_fam, fixed_base, observational, description in comparisons_spec:
        base_all = combined[combined.model_family == base_fam]
        treat_all = combined[combined.model_family == treat_fam]
        if base_all.empty or treat_all.empty:
            continue
        for stratum_name, metric_specs in stratum_metrics.items():
            stratum_panel = strata[stratum_name]
            for metric, filter_col in metric_specs:
                if filter_col is not None:
                    eligible_ids = set(ref_df.loc[ref_df[filter_col], "sample_id"].astype(str))
                    target_panel = stratum_panel[stratum_panel.sample_id.isin(eligible_ids)]
                else:
                    target_panel = stratum_panel
                if len(target_panel) < 2 or target_panel["cluster_id"].nunique() < 2:
                    continue
                target_ids = set(target_panel["sample_id"].astype(str))
                res = _paired_summary_with_p(
                    base_all[base_all.sample_id.isin(target_ids)],
                    treat_all[treat_all.sample_id.isin(target_ids)],
                    target_panel[["sample_id", "cluster_id", "prompt"]],
                    metric=metric,
                    decode_seeds=decode_seeds,
                    fixed_baseline=fixed_base,
                    iters=iters,
                )
                res["comparison"] = comp_name
                res["stratum"] = stratum_name
                res["observational"] = observational
                res["description"] = description
                res["holm_p_value"] = None
                paired_dict[f"{comp_name}:{stratum_name}"][metric] = res

    # Holm-Bonferroni adjustment on primary main-panel endpoint families per comparison.
    for comp_name, _, _, _, _, _ in comparisons_spec:
        bucket = paired_dict.get(f"{comp_name}:main", {})
        present_primary = [m for m in PRIMARY_HOLM_METRICS if m in bucket]
        adjusted = holm_adjust([bucket[m]["p_value"] for m in present_primary])
        for metric, adj_p in zip(present_primary, adjusted):
            bucket[metric]["holm_p_value"] = adj_p

    for stratum_key, metrics_map in paired_dict.items():
        for metric, res in metrics_map.items():
            paired_rows.append({
                "comparison": res["comparison"],
                "stratum": res["stratum"],
                "metric": metric,
                "n_prompts": res["n_prompts"],
                "n_clusters": res["n_clusters"],
                "baseline_fixed": res["baseline_fixed"],
                "observational": res["observational"],
                "baseline_mean": res["baseline_mean"],
                "treatment_mean": res["treatment_mean"],
                "delta": res["delta"],
                "ci95_low": res["paired_cluster_training_seed_ci95"][0],
                "ci95_high": res["paired_cluster_training_seed_ci95"][1],
                "p_value": res["p_value"],
                "holm_p_value": res["holm_p_value"],
            })

    paired_df = pd.DataFrame(paired_rows)
    category_tag_df = pd.DataFrame(_build_category_tag_rows(combined, ref_df, main_panel))
    per_token_df = _build_per_token_df(combined, ref_df, main_panel)

    summary_payload = {
        "panel_manifest": panel_manifest,
        "decode_seeds": decode_seeds,
        "bootstrap_iters": iters,
        "reference_control": ref_summary,
        "model_summaries": model_summaries,
        "best_of_k": bok_summary,
        "best_of_k_paired": {k: v for k, v in bok_paired.items()},
        "paired_comparisons": {k: v for k, v in paired_dict.items()},
        "limitations": [
            "AMRFinderPlus matches and complete predicted ORFs do not establish expressed antibiotic resistance.",
            "Unrequested ORI/AMR measures distinct detected token classes, not a count of physical cassette copies.",
            "COPY_LOW is assessed through origin-family compatibility proxies and conflicting-origin detections, not measured plasmid copy number.",
            "Experiment 2 compares 3 GRPO-v2 seeds initialized from a single cluster-disjoint pretrained base (no pretraining-seed replication).",
        ],
    }

    if output_dir is not None:
        _write_outputs(
            output_dir=output_dir,
            summary_payload=summary_payload,
            combined_df=combined,
            ref_df=ref_df,
            paired_df=paired_df,
            ref_amr_cov_df=ref_amr_cov_df,
            category_tag_df=category_tag_df,
            per_token_df=per_token_df,
            model_summaries=model_summaries,
        )
    return summary_payload


def _build_per_token_df(
    combined_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    main_panel: pd.DataFrame,
) -> pd.DataFrame:
    main_ids = set(main_panel["sample_id"].astype(str))
    rows_out: list[dict[str, Any]] = []
    sources: list[tuple[str, pd.DataFrame]] = [("reference_control", ref_df)]
    for fam in MODEL_FAMILY_SPECS:
        fam_df = combined_df[combined_df.model_family == fam]
        if not fam_df.empty:
            sources.append((fam, fam_df))
    for fam_name, df in sources:
        sub = df[df.sample_id.isin(main_ids)]
        req_counts: collections.Counter = collections.Counter()
        hit_counts: collections.Counter = collections.Counter()
        bg_counts: collections.Counter = collections.Counter()
        n_rows = len(sub)
        for row in sub.itertuples():
            req = set(json.loads(row.requested_tokens_json))
            det = set(json.loads(row.detected_tokens_json))
            for tok in req:
                req_counts[tok] += 1
                hit_counts[tok] += int(tok in det)
            for tok in det - req:
                bg_counts[tok] += 1
        for tok in sorted(req_counts):
            req_n = req_counts[tok]
            non_req_n = max(n_rows - req_n, 1)
            hit_rate = hit_counts[tok] / req_n
            bg_rate = bg_counts[tok] / non_req_n
            rows_out.append({
                "model_family": fam_name,
                "token": tok,
                "category": category_of(tok),
                "requested": int(req_n),
                "hits": int(hit_counts[tok]),
                "hit_rate": hit_rate,
                "unrequested_hits": int(bg_counts[tok]),
                "unrequested_rate": bg_rate,
                "lift": hit_rate - bg_rate,
            })
    return pd.DataFrame(rows_out)


def _format_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in df.to_dict("records"):
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append("—" if math.isnan(val) else f"{val:.4f}")
            elif val is None:
                cells.append("—")
            else:
                cells.append(str(val))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def _write_outputs(
    *,
    output_dir: Path,
    summary_payload: dict[str, Any],
    combined_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    ref_amr_cov_df: pd.DataFrame,
    category_tag_df: pd.DataFrame,
    per_token_df: pd.DataFrame,
    model_summaries: dict[str, Any],
) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    all_obs = pd.concat([ref_df, combined_df], ignore_index=True)
    all_obs.to_parquet(output_dir / "all_observations.parquet", index=False)
    paired_df.to_csv(tables_dir / "paired_comparisons.csv", index=False)
    ref_amr_cov_df.to_csv(tables_dir / "reference_control_amr_coverage.csv", index=False)
    category_tag_df.to_csv(tables_dir / "category_and_tag_breakdown.csv", index=False)
    per_token_df.to_csv(tables_dir / "per_token_breakdown.csv", index=False)

    main_rows = []
    for fam, info in model_summaries["main"].items():
        row = {"model_family": fam, "n_prompts": info["n_prompts"], "n_training_seeds": info["n_training_seeds"]}
        for metric, stats in info["metrics"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        main_rows.append(row)
    pd.DataFrame(main_rows).to_csv(tables_dir / "main_panel_summary.csv", index=False)

    minority_rows = []
    for stratum in model_summaries:
        if not stratum.startswith("minority_"):
            continue
        for fam, info in model_summaries[stratum].items():
            row = {
                "stratum": stratum,
                "model_family": fam,
                "n_prompts": info["n_prompts"],
                "n_training_seeds": info["n_training_seeds"],
            }
            for metric, stats in info["metrics"].items():
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            minority_rows.append(row)
    pd.DataFrame(minority_rows).to_csv(tables_dir / "minority_panels_summary.csv", index=False)

    copy_rows = [r for r in minority_rows if r["stratum"] == "minority_copy_low"]
    pd.DataFrame(copy_rows).to_csv(tables_dir / "copy_low_breakdown.csv", index=False)

    def _pct(val: float | None) -> str:
        return f"{val:.3%}" if val is not None else "n/a"

    ref_info = summary_payload["reference_control"]
    lines = [
        "# Rigor Evaluation Summary (`frozen_panel_v3`)",
        "",
        f"- Decode seeds averaged per model: `{summary_payload['decode_seeds']}`",
        f"- Reference control (`n={ref_info['n_main']}`): "
        f"viable={_pct(ref_info['viable'])}, "
        f"mean_F={_pct(ref_info['faithfulness'])}, "
        f"useful@0.5={_pct(ref_info['useful_05'])}, "
        f"AMRFinder intact recall={_pct(ref_info['amr_orthogonal_intact_recall'])}, "
        f"Prodigal peptide tag recall={_pct(ref_info['tag_recall_prodigal_peptide'])}",
        "",
        "## Primary Paired Comparisons (`main` panel)",
        "",
    ]
    main_paired = paired_df[paired_df.stratum == "main"]
    if not main_paired.empty:
        lines.append(_format_markdown_table(main_paired[[
            "comparison", "metric", "baseline_mean", "treatment_mean",
            "delta", "ci95_low", "ci95_high", "p_value", "holm_p_value",
        ]]))
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=10000)
    args = parser.parse_args()

    panel_df = pd.read_parquet(args.panel)
    panel_manifest = json.loads(args.manifest.read_text())
    ref_df = pd.read_parquet(args.reference)
    frames = [pd.read_parquet(path) for path in args.metrics]
    summarize_rigor_evaluation(
        panel_df=panel_df,
        panel_manifest=panel_manifest,
        ref_df=ref_df,
        model_frames=frames,
        iters=args.iters,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
