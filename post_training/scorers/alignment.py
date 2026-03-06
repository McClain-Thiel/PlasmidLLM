"""Smith-Waterman alignment scorer using score-ratio metric.

Reward = sum of per-component scores, where each component contributes up to 1.0.
Score-ratio = sw_score / self_alignment_score, naturally captures both identity
and coverage. For CDS motifs, tries both DNA and protein alignment (6-frame
translation) and takes the max.

This is the scorer used in GRPO training runs.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import parasail
from Bio.Seq import Seq

from post_training.scorers.base import Scorer

# ── Config ────────────────────────────────────────────────────────────────────

DNA_MATRIX = parasail.matrix_create("ACGT", 1, -1)
PROTEIN_MATRIX = parasail.blosum62
DNA_OPEN, DNA_EXTEND = 5, 1
PROTEIN_OPEN, PROTEIN_EXTEND = 10, 1

CDS_CATEGORIES = {"AMR", "REPORTER", "TAG"}
HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}
QC_THRESHOLD = 0.70

EXCLUDE_TOKENS = {"<ELEM_IRES>", "<ELEM_TRACRRNA>"}
MAX_CATEGORY_REPRESENTATIVES = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_translate(dna_seq: str) -> Optional[str]:
    """Translate DNA to protein, trimming to codon boundary."""
    if not dna_seq or len(dna_seq) < 3:
        return None
    try:
        trimmed = dna_seq[: len(dna_seq) - (len(dna_seq) % 3)]
        return str(Seq(trimmed).translate())
    except Exception:
        return None


def _compute_max_score(seq: str, is_protein: bool) -> int:
    """Self-alignment score for normalization."""
    if not seq or len(seq) == 0:
        return 1
    if is_protein:
        return parasail.sw_striped_16(
            seq, seq, PROTEIN_OPEN, PROTEIN_EXTEND, PROTEIN_MATRIX
        ).score
    else:
        return parasail.sw_striped_16(
            seq, seq, DNA_OPEN, DNA_EXTEND, DNA_MATRIX
        ).score


def load_motif_lookup(path: str) -> pd.DataFrame:
    """Load motif lookup from parquet into a pandas DataFrame.

    Adds computed columns:
      - protein_seq (translated from dna_seq for CDS tokens where seq_type != 'protein')
      - dna_max_score, protein_max_score (self-alignment scores)
    """
    df = pd.read_parquet(path)

    if "is_cds" not in df.columns and "category" in df.columns:
        df["is_cds"] = df["category"].isin(CDS_CATEGORIES)

    if "dna_seq" not in df.columns and "sequence" in df.columns:
        df["dna_seq"] = df.apply(
            lambda r: r["sequence"] if pd.notna(r.get("seq_type")) and r["seq_type"] != "protein" else None,
            axis=1,
        )
        df["protein_seq"] = df.apply(
            lambda r: r["sequence"] if r.get("seq_type") == "protein" else None,
            axis=1,
        )

    mask_cds_dna = df["is_cds"] & df["dna_seq"].notna() & (df["seq_type"] != "protein")
    df.loc[mask_cds_dna, "protein_seq"] = df.loc[mask_cds_dna, "dna_seq"].apply(
        safe_translate
    )

    df["dna_max_score"] = df["dna_seq"].apply(
        lambda s: _compute_max_score(s, is_protein=False) if pd.notna(s) else 1
    )
    df["protein_max_score"] = df["protein_seq"].apply(
        lambda s: _compute_max_score(s, is_protein=True) if pd.notna(s) else 1
    )

    df = df.set_index("token", drop=False)
    df.index.name = "token_idx"

    return df


def _extract_category(token: str) -> Optional[str]:
    """Extract category prefix from a hard token, e.g. '<ORI_COLE1>' -> 'ORI'."""
    inner = token.strip("<>")
    for prefix in HARD_PREFIXES:
        if inner.startswith(prefix):
            return prefix.rstrip("_")
    return None


def build_category_index(
    lookup_df: pd.DataFrame, max_per_category: int = MAX_CATEGORY_REPRESENTATIVES
) -> dict:
    """Pre-group lookup entries by category with a capped representative subset."""
    if "category" not in lookup_df.columns:
        lookup_df = lookup_df.copy()
        lookup_df["category"] = lookup_df["token"].apply(
            lambda t: _extract_category(t)
        )

    index = {}
    for cat, group in lookup_df.groupby("category"):
        if cat is None:
            continue
        unique_tokens = group["token"].unique()
        if len(unique_tokens) <= max_per_category:
            reps = group.groupby("token").first().reset_index()
        else:
            sampled = np.random.RandomState(42).choice(
                unique_tokens, size=max_per_category, replace=False
            )
            reps = group[group["token"].isin(sampled)].groupby("token").first().reset_index()
        index[cat] = reps
    return index


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_dna_score(motif_dna: str, candidate_seq: str, max_score: int) -> float:
    """DNA Smith-Waterman on both strands. Returns score_ratio in [0, 1]."""
    if not motif_dna or len(candidate_seq) == 0:
        return 0.0

    motif_rev = str(Seq(motif_dna).reverse_complement())

    score_fwd = parasail.sw_striped_16(
        candidate_seq, motif_dna, DNA_OPEN, DNA_EXTEND, DNA_MATRIX
    ).score
    score_rev = parasail.sw_striped_16(
        candidate_seq, motif_rev, DNA_OPEN, DNA_EXTEND, DNA_MATRIX
    ).score

    best = max(score_fwd, score_rev)
    return round(min(best / max(max_score, 1), 1.0), 4)


def align_protein_score(
    motif_protein: str, candidate_seq: str, max_score: int
) -> float:
    """Protein Smith-Waterman via 6-frame translation. Returns score_ratio."""
    if not motif_protein or len(candidate_seq) < 3:
        return 0.0

    fwd = candidate_seq.upper()
    rev = str(Seq(fwd).reverse_complement())

    best_score = 0
    for offset in range(3):
        for seq in [fwd, rev]:
            sub = seq[offset:]
            sub = sub[: len(sub) - (len(sub) % 3)]
            if len(sub) < 3:
                continue
            try:
                prot = str(Seq(sub).translate())
            except Exception:
                continue
            if not prot:
                continue

            score = parasail.sw_striped_16(
                prot, motif_protein, PROTEIN_OPEN, PROTEIN_EXTEND, PROTEIN_MATRIX
            ).score
            best_score = max(best_score, score)

    return round(min(best_score / max(max_score, 1), 1.0), 4)


# ── Token parsing ─────────────────────────────────────────────────────────────

def parse_hard_tokens(prompt: str, lookup_df: pd.DataFrame) -> list[str]:
    """Extract hard tokens from prompt that exist in the lookup."""
    all_tokens = re.findall(r"<[^>]+>", prompt)
    known = set(lookup_df.index.unique())
    hard = []
    for t in all_tokens:
        inner = t.strip("<>")
        if any(inner.startswith(p) for p in HARD_PREFIXES):
            if t in known and t not in EXCLUDE_TOKENS:
                hard.append(t)
    return hard


def score_motif(
    token: str, candidate_seq: str, lookup_df: pd.DataFrame
) -> dict:
    """Score one motif token against a candidate sequence."""
    if token not in lookup_df.index:
        return {"token": token, "score_ratio": 0.0, "found": False}

    rows = lookup_df.loc[[token]]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T

    best_ratio = 0.0

    for _, row in rows.iterrows():
        if pd.notna(row.get("dna_seq")):
            ratio = align_dna_score(
                row["dna_seq"], candidate_seq, int(row["dna_max_score"])
            )
            best_ratio = max(best_ratio, ratio)

        if row.get("is_cds") and pd.notna(row.get("protein_seq")):
            ratio = align_protein_score(
                row["protein_seq"], candidate_seq, int(row["protein_max_score"])
            )
            best_ratio = max(best_ratio, ratio)

    return {
        "token": token,
        "score_ratio": best_ratio,
        "found": best_ratio >= QC_THRESHOLD,
    }


# ── Scorer class ──────────────────────────────────────────────────────────────

class AlignmentScorer(Scorer):
    """Smith-Waterman alignment scorer.

    Reward = sum of per-component scores, each capped at 1.0 via
    min(1.0, score_ratio / QC_THRESHOLD). Plus optional EOS bonus.
    """

    def __init__(
        self,
        motif_lookup_path: str | None = None,
        lookup_df: pd.DataFrame | None = None,
        eos_bonus: float = 0.15,
    ):
        if lookup_df is not None:
            self.lookup_df = lookup_df
        elif motif_lookup_path is not None:
            self.lookup_df = load_motif_lookup(motif_lookup_path)
        else:
            raise ValueError("Must provide either motif_lookup_path or lookup_df")
        self.eos_bonus = eos_bonus

    def _extract_dna(self, text: str) -> tuple[str, bool]:
        """Strip tokens, return (clean_dna, has_eos)."""
        raw = text.upper()
        has_eos = "<EOS>" in text or "</s>" in text
        seq = re.sub(r"<[^>]+>", "", raw)
        seq = re.sub(r"[^ATGCN]", "", seq)
        return seq, has_eos

    def score_sequence(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> float:
        seq, has_eos = self._extract_dna(sequence)
        if len(seq) < 20:
            return 0.0

        hard_tokens = parse_hard_tokens(prompt, self.lookup_df)
        if not hard_tokens:
            return 0.0

        component_scores = []
        for token in hard_tokens:
            result = score_motif(token, seq, self.lookup_df)
            component_scores.append(min(1.0, result["score_ratio"] / QC_THRESHOLD))

        reward = float(np.sum(component_scores))
        return reward + (self.eos_bonus if has_eos else 0.0)

    def score_sequence_detailed(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> dict[str, Any]:
        seq, has_eos = self._extract_dna(sequence)
        hard_tokens = parse_hard_tokens(prompt, self.lookup_df)

        if not hard_tokens or len(seq) < 20:
            return {
                "reward": 0.0,
                "n_hard_tokens": len(hard_tokens),
                "n_found": 0,
                "qc_pass_rate": 0.0,
                "per_motif": [],
            }

        per_motif = []
        component_scores = []

        for token in hard_tokens:
            motif_result = score_motif(token, seq, self.lookup_df)
            per_motif.append(motif_result)
            component_scores.append(min(1.0, motif_result["score_ratio"] / QC_THRESHOLD))

        n_found = sum(1 for m in per_motif if m["found"])
        reward = float(np.sum(component_scores))
        reward += self.eos_bonus if has_eos else 0.0

        return {
            "reward": round(reward, 4),
            "n_hard_tokens": len(hard_tokens),
            "n_found": n_found,
            "qc_pass_rate": round(n_found / len(per_motif), 4),
            "per_motif": per_motif,
            "component_scores": [round(s, 4) for s in component_scores],
        }
