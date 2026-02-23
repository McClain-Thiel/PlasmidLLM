"""Reward function for PlasmidLM post-training using sequence alignment."""

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import parasail
from Bio.Seq import Seq

# ── Config ────────────────────────────────────────────────────────────────────

DNA_MATRIX = parasail.matrix_create("ACGT", 1, -1)
PROTEIN_MATRIX = parasail.blosum62
DNA_OPEN, DNA_EXTEND = 5, 1
PROTEIN_OPEN, PROTEIN_EXTEND = 10, 1

CDS_CATEGORIES = {"AMR", "REPORTER", "TAG"}
HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}
QC_THRESHOLD = 0.70

# Tokens with known poor registry coverage — exclude from reward by default
EXCLUDE_TOKENS = {"<ELEM_IRES>", "<ELEM_TRACRRNA>"}


# ── Loading ───────────────────────────────────────────────────────────────────

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
    """
    Load motif lookup from parquet into a pandas DataFrame.

    Adds computed columns:
      - protein_seq (translated from dna_seq for CDS tokens where seq_type != 'protein')
      - dna_max_score, protein_max_score (self-alignment scores)

    Args:
        path: Path to parquet file or directory

    Returns:
        DataFrame with columns: token, sseqid, category, is_cds, seq_type,
        dna_seq, protein_seq, dna_max_score, protein_max_score
    """
    df = pd.read_parquet(path)

    # Derive is_cds from category (motif_registry.parquet uses 'category' not 'is_cds')
    if "is_cds" not in df.columns and "category" in df.columns:
        df["is_cds"] = df["category"].isin(CDS_CATEGORIES)

    # Split 'sequence' column into dna_seq / protein_seq based on seq_type
    if "dna_seq" not in df.columns and "sequence" in df.columns:
        df["dna_seq"] = df.apply(
            lambda r: r["sequence"] if pd.notna(r.get("seq_type")) and r["seq_type"] != "protein" else None,
            axis=1,
        )
        df["protein_seq"] = df.apply(
            lambda r: r["sequence"] if r.get("seq_type") == "protein" else None,
            axis=1,
        )

    # For CDS tokens with DNA sequences, compute protein translations
    mask_cds_dna = df["is_cds"] & df["dna_seq"].notna() & (df["seq_type"] != "protein")
    df.loc[mask_cds_dna, "protein_seq"] = df.loc[mask_cds_dna, "dna_seq"].apply(
        safe_translate
    )

    # Compute max scores
    df["dna_max_score"] = df["dna_seq"].apply(
        lambda s: _compute_max_score(s, is_protein=False) if pd.notna(s) else 1
    )
    df["protein_max_score"] = df["protein_seq"].apply(
        lambda s: _compute_max_score(s, is_protein=True) if pd.notna(s) else 1
    )

    # Index by token for fast groupby lookups
    df = df.set_index("token", drop=False)
    df.index.name = "token_idx"

    return df


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


# ── Scoring ───────────────────────────────────────────────────────────────────

def parse_hard_tokens(prompt: str, lookup_df: pd.DataFrame) -> List[str]:
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
    """
    Score one motif token against a candidate sequence.
    Tries all variants (rows) for that token, returns best score.
    """
    if token not in lookup_df.index:
        return {"token": token, "score_ratio": 0.0, "found": False}

    rows = lookup_df.loc[[token]]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T

    best_ratio = 0.0

    for _, row in rows.iterrows():
        # DNA alignment
        if pd.notna(row.get("dna_seq")):
            ratio = align_dna_score(
                row["dna_seq"], candidate_seq, int(row["dna_max_score"])
            )
            best_ratio = max(best_ratio, ratio)

        # Protein alignment (CDS only)
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


def compute_reward(
    prompt: str,
    candidate_seq: str,
    lookup_df: pd.DataFrame,
    return_details: bool = False,
) -> float | Tuple[float, dict]:
    """
    Compute scalar reward for a generated sequence.

    Reward = mean of per-motif score_ratios across all hard tokens.
    """
    candidate_seq = candidate_seq.upper().strip()
    hard_tokens = parse_hard_tokens(prompt, lookup_df)

    if not hard_tokens:
        if return_details:
            return 0.0, {"reward": 0.0, "n_hard_tokens": 0, "n_found": 0,
                         "qc_pass_rate": 0.0, "per_motif": []}
        return 0.0

    per_motif = [score_motif(t, candidate_seq, lookup_df) for t in hard_tokens]
    scores = [m["score_ratio"] for m in per_motif]
    n_found = sum(1 for m in per_motif if m["found"])
    reward = float(np.mean(scores))

    if return_details:
        return reward, {
            "reward": round(reward, 4),
            "n_hard_tokens": len(hard_tokens),
            "n_found": n_found,
            "qc_pass_rate": round(n_found / len(per_motif), 4),
            "per_motif": per_motif,
        }
    return reward


# ── Batch API (GRPO/PPO compatible) ──────────────────────────────────────────

def plasmid_reward_fn(
    prompts: List[str],
    completions: List[str],
    lookup_df: pd.DataFrame,
) -> List[float]:
    """
    Batch reward function for RL training.

    Args:
        prompts: List of prompt strings
        completions: List of generated DNA sequences
        lookup_df: Loaded motif lookup DataFrame

    Returns:
        List of reward floats in [0, 1]
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        seq = completion.upper()
        # Strip any special tokens from completion
        seq = re.sub(r"<[^>]+>", "", seq)
        seq = re.sub(r"[^ATGCN]", "", seq)

        if len(seq) < 100:
            rewards.append(0.0)
            continue

        rewards.append(compute_reward(prompt, seq, lookup_df))

    return rewards
