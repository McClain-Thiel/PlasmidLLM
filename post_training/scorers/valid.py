"""Valid plasmid scorer — curriculum-learning reward with structural penalties.

Wraps the PlannotateScorer (BLAST-based element detection) and adds
biological-validity penalties/bonuses on top:

Penalties
---------
* **Excess ORIs** — real plasmids almost never have >1 origin of replication.
* **Excess AMRs** — >2 resistance genes is unusual and a sign of element stuffing.
* **Unrequested elements** — functional elements (AMR, ORI, REPORTER, TAG)
  detected via BLAST that were NOT in the prompt.
* **Length** — gentle penalty for sequences longer than ~8 kbp; most useful
  cloning vectors are 4–8 kbp.

Bonuses
-------
* **Promoter–CDS adjacency** — when a requested promoter and its matching CDS
  are both found AND are within ``adjacency_max_gap`` bp of each other on the
  query, a small bonus is applied.

The final reward is::

    final = clip(base_plannotate_reward * (1 + bonuses − penalties), 0, 1)

The multiplicative formulation means penalties only bite when the base score
is nonzero (garbage sequences that already score 0 aren't further penalised).
All weights are constructor parameters for easy tuning via config.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

import pandas as pd

from post_training.scorers.base import Scorer
from post_training.scorers.plannotate import (
    HARD_PREFIXES,
    PlannotateScorer,
    _clean_dna,
    _parse_hard_tokens,
    _sanitize_id,
)


# ── Category helpers ──────────────────────────────────────────────────────────

def _extract_category(token: str) -> str | None:
    """'<AMR_AMPICILLIN>' -> 'AMR'."""
    inner = token.strip("<>")
    for prefix in HARD_PREFIXES:
        if inner.startswith(prefix):
            return prefix.rstrip("_")
    return None


# Categories where unrequested detections trigger a penalty.
# Promoters are excluded because they legitimately accompany CDS elements
# even when not explicitly requested (e.g. AmpR promoter with AmpR gene).
PENALISED_UNREQUESTED_CATEGORIES = {"AMR", "ORI", "REPORTER", "TAG"}

# Promoter → CDS pairs that should be adjacent.
# ``None`` means the promoter can drive *any* CDS (general-purpose promoter).
_ANY_CDS = None
PROMOTER_CDS_AFFINITY: dict[str, set[str] | None] = {
    "<PROM_AMPR>": {"<AMR_AMPICILLIN>"},
    "<PROM_LAC>": _ANY_CDS,
    "<PROM_CMV>": _ANY_CDS,
    "<PROM_T7>": _ANY_CDS,
    "<PROM_T5>": _ANY_CDS,
    "<PROM_T3>": set(),  # in vitro transcription, not CDS driver
    "<PROM_SV40>": _ANY_CDS,
    "<PROM_U6>": set(),  # drives gRNAs, not CDS
    "<PROM_EF1A>": _ANY_CDS,
    "<PROM_RSV>": _ANY_CDS,
    "<PROM_SP6>": set(),  # in vitro transcription, not CDS driver
    "<PROM_CAG>": _ANY_CDS,
}

CDS_CATEGORIES = {"AMR", "REPORTER", "TAG"}
ORI_LOCUS_MIN_OVERLAP = 0.5


def _hit_bounds(hit: dict[str, Any]) -> tuple[int, int]:
    """Return ordered query coordinates for a BLAST hit."""
    return min(hit["qstart"], hit["qend"]), max(hit["qstart"], hit["qend"])


def _overlap_fraction(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Overlap relative to the smaller interval."""
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    span_a = max(1, a[1] - a[0])
    span_b = max(1, b[1] - b[0])
    return overlap / min(span_a, span_b)


def _collapse_ori_loci(
    broad_hits: dict[str, dict[str, Any]],
    min_overlap: float = ORI_LOCUS_MIN_OVERLAP,
) -> list[dict[str, Any]]:
    """Merge overlapping ORI token hits into physical origin loci."""
    ori_hits = [
        (token, hit) for token, hit in broad_hits.items()
        if _extract_category(token) == "ORI"
    ]
    if not ori_hits:
        return []

    clusters: list[dict[str, Any]] = []
    for token, hit in sorted(ori_hits, key=lambda item: _hit_bounds(item[1])[0]):
        bounds = _hit_bounds(hit)
        target_cluster: dict[str, Any] | None = None
        for cluster in clusters:
            if any(
                _overlap_fraction(bounds, member["bounds"]) >= min_overlap
                for member in cluster["members"]
            ):
                target_cluster = cluster
                break

        member = {"token": token, "hit": hit, "bounds": bounds}
        if target_cluster is None:
            clusters.append({"members": [member]})
        else:
            target_cluster["members"].append(member)

    loci: list[dict[str, Any]] = []
    for cluster in clusters:
        members = cluster["members"]
        best = max(members, key=lambda member: member["hit"]["bit_score"])
        loci.append(
            {
                "rep_token": best["token"],
                "tokens": {member["token"] for member in members},
                "bounds": (
                    min(member["bounds"][0] for member in members),
                    max(member["bounds"][1] for member in members),
                ),
            }
        )
    return loci


# ── BLAST output parser with query positions ──────────────────────────────────

def _parse_broad_hits(
    tsv_path: str,
    sseqid_to_token: dict[str, str],
    fasta_id_to_sseqid: dict[str, str],
    min_coverage: float = 30.0,
    min_identity: float = 0.0,
) -> dict[str, dict[str, dict]]:
    """Parse BLAST tabular output keeping *all* significant token hits with positions.

    Returns ``{query_id: {token: best_hit_info}}``.
    Each hit_info includes ``qstart`` and ``qend`` for adjacency checking.
    """
    results: dict[str, dict[str, dict]] = {}

    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        return results

    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 14:
                continue

            qseqid = parts[0]
            raw_sseqid = parts[1]
            pident = float(parts[2])
            aln_len = int(parts[3])
            qstart = int(parts[6])
            qend = int(parts[7])
            evalue = float(parts[10])
            bitscore = float(parts[11])
            nident = int(parts[12])
            slen = int(parts[13])

            token = _resolve_broad(raw_sseqid, sseqid_to_token, fasta_id_to_sseqid)
            if token is None:
                continue

            coverage = min((aln_len / max(slen, 1)) * 100, 100.0)
            if coverage < min_coverage:
                continue
            if pident < min_identity:
                continue

            norm_score = bitscore / max(slen, 1)

            hit_info = {
                "token": token,
                "sseqid": raw_sseqid,
                "bit_score": bitscore,
                "evalue": evalue,
                "pct_id": round(min(pident, 100.0), 2),
                "coverage": round(coverage, 2),
                "norm_score": round(norm_score, 4),
                "identity": nident,
                "alignment_len": aln_len,
                "target_len": slen,
                "qstart": qstart,
                "qend": qend,
            }

            per_token = results.setdefault(qseqid, {})
            prev = per_token.get(token)
            if prev is None or hit_info["bit_score"] > prev["bit_score"]:
                per_token[token] = hit_info

    return results


def _resolve_broad(
    raw_sseqid: str,
    sseqid_to_token: dict[str, str],
    fasta_id_to_sseqid: dict[str, str],
) -> str | None:
    cid = raw_sseqid.rsplit("|", 1)[-1] if "|" in raw_sseqid else raw_sseqid
    if cid in sseqid_to_token:
        return sseqid_to_token[cid]
    orig = fasta_id_to_sseqid.get(cid)
    if orig and orig in sseqid_to_token:
        return sseqid_to_token[orig]
    return None


# ── Scorer ────────────────────────────────────────────────────────────────────

class ValidPlasmidScorer(Scorer):
    """Curriculum-stage scorer: plannotate base reward + structural validity.

    Parameters
    ----------
    excess_ori_penalty : float
        Penalty per ORI beyond the first detected.
    excess_amr_penalty : float
        Penalty per AMR beyond the second detected.
    unrequested_penalty : float
        Penalty per unrequested functional element (AMR/ORI/REPORTER/TAG).
    length_penalty_per_kb : float
        Penalty per kbp over ``length_soft_max``.
    length_soft_max : float
        Upper end of the ideal size range (bp).
    length_penalty_cap : float
        Maximum total length penalty.
    adjacency_bonus : float
        Bonus per correctly-adjacent promoter–CDS pair.
    adjacency_max_gap : int
        Maximum gap (bp) for a promoter–CDS pair to count as adjacent.
    broad_min_coverage : float
        Minimum coverage (%) for a broad-scan hit to count.
    broad_min_identity : float
        Minimum percent identity for a broad-scan hit to count.
    """

    def __init__(
        self,
        # --- PlannotateScorer pass-through ---
        plannotate_db_path: str | None = None,
        plannotate_df: pd.DataFrame | None = None,
        motif_registry_path: str | None = None,
        motif_registry_df: pd.DataFrame | None = None,
        db_dir: str | None = None,
        evalue: float = 1e-5,
        filter_db: bool = True,
        # --- Structural penalty weights ---
        excess_ori_penalty: float = 0.15,
        excess_amr_penalty: float = 0.10,
        unrequested_penalty: float = 0.05,
        length_penalty_per_kb: float = 0.01,
        length_soft_max: float = 8000.0,
        length_penalty_cap: float = 0.15,
        adjacency_bonus: float = 0.05,
        adjacency_max_gap: int = 500,
        broad_min_coverage: float = 30.0,
        broad_min_identity: float = 0.0,
    ):
        self._base = PlannotateScorer(
            plannotate_db_path=plannotate_db_path,
            plannotate_df=plannotate_df,
            motif_registry_path=motif_registry_path,
            motif_registry_df=motif_registry_df,
            db_dir=db_dir,
            evalue=evalue,
            filter_db=filter_db,
        )

        self.excess_ori_penalty = excess_ori_penalty
        self.excess_amr_penalty = excess_amr_penalty
        self.unrequested_penalty = unrequested_penalty
        self.length_penalty_per_kb = length_penalty_per_kb
        self.length_soft_max = length_soft_max
        self.length_penalty_cap = length_penalty_cap
        self.adjacency_bonus_weight = adjacency_bonus
        self.adjacency_max_gap = adjacency_max_gap
        self.broad_min_coverage = broad_min_coverage
        self.broad_min_identity = broad_min_identity

        self._all_sseqid_to_token = self._build_full_token_map()

    # ------------------------------------------------------------------ init
    def _build_full_token_map(self) -> dict[str, str]:
        """Reverse map: sseqid → token for every token in the bridge."""
        m: dict[str, str] = {}
        for token, sseqids in self._base._token_bridge.items():
            for sid in sseqids:
                m[_sanitize_id(sid)] = token
                m[sid] = token
        return m

    # --------------------------------------------------------------- BLAST
    def _run_broad_blast(
        self,
        query_seqs: list[tuple[str, str]],
    ) -> dict[str, dict[str, dict]]:
        """Run BLAST against the full DB and parse with all-token mapping + positions."""
        if not query_seqs:
            return {}

        query_fasta = os.path.join(self._base.db_dir, "valid_query.fasta")
        with open(query_fasta, "w") as f:
            for qid, seq in query_seqs:
                f.write(f">{qid}\n{seq}\n")

        merged: dict[str, dict[str, dict]] = {}

        if self._base._nucl_db:
            out_tsv = os.path.join(self._base.db_dir, "valid_blastn.tsv")
            self._base._run_blast_cmd(
                "blastn", query_fasta, self._base._nucl_db, out_tsv,
                evalue_override=10.0,
                extra_args=["-word_size", "7"],
            )
            hits = _parse_broad_hits(
                out_tsv, self._all_sseqid_to_token,
                self._base._fasta_id_to_sseqid,
                min_coverage=self.broad_min_coverage,
                min_identity=self.broad_min_identity,
            )
            for qid, per_token in hits.items():
                dest = merged.setdefault(qid, {})
                for tok, info in per_token.items():
                    prev = dest.get(tok)
                    if prev is None or info["bit_score"] > prev["bit_score"]:
                        dest[tok] = info

        if self._base._prot_db:
            out_tsv = os.path.join(self._base.db_dir, "valid_blastx.tsv")
            self._base._run_blast_cmd(
                "blastx", query_fasta, self._base._prot_db, out_tsv,
            )
            hits = _parse_broad_hits(
                out_tsv, self._all_sseqid_to_token,
                self._base._fasta_id_to_sseqid,
                min_coverage=self.broad_min_coverage,
                min_identity=self.broad_min_identity,
            )
            for qid, per_token in hits.items():
                dest = merged.setdefault(qid, {})
                for tok, info in per_token.items():
                    prev = dest.get(tok)
                    if prev is None or info["bit_score"] > prev["bit_score"]:
                        dest[tok] = info

        return merged

    # --------------------------------------------------------- penalties
    def _compute_structural(
        self,
        expected_tokens: list[str],
        broad_hits: dict[str, dict],
        seq_len: int,
    ) -> dict[str, Any]:
        """Compute all structural penalties and bonuses.

        Returns a dict with the multiplier and a per-component breakdown.
        """
        expected_set = set(expected_tokens)
        found_tokens = set(broad_hits.keys())
        ori_loci = _collapse_ori_loci(broad_hits)
        expected_ori_tokens = {t for t in expected_set if _extract_category(t) == "ORI"}

        # ── Excess ORIs ───────────────────────────────────────────────
        ori_found = {t for locus in ori_loci for t in locus["tokens"]}
        n_ori = len(ori_loci)
        ori_pen = max(0, n_ori - 1) * self.excess_ori_penalty

        # ── Excess AMRs ──────────────────────────────────────────────
        amr_found = {t for t in found_tokens if _extract_category(t) == "AMR"}
        n_amr = len(amr_found)
        amr_pen = max(0, n_amr - 2) * self.excess_amr_penalty

        # ── Unrequested functional elements ──────────────────────────
        unrequested = []
        for tok in found_tokens - expected_set:
            cat = _extract_category(tok)
            if cat == "ORI":
                continue
            if cat in PENALISED_UNREQUESTED_CATEGORIES:
                unrequested.append(tok)
        for locus in ori_loci:
            if locus["tokens"] & expected_ori_tokens:
                continue
            unrequested.append(locus["rep_token"])
        unreq_pen = len(unrequested) * self.unrequested_penalty

        # ── Length penalty ────────────────────────────────────────────
        excess_kb = max(0.0, (seq_len - self.length_soft_max) / 1000.0)
        len_pen = min(excess_kb * self.length_penalty_per_kb, self.length_penalty_cap)

        # ── Promoter–CDS adjacency bonus ─────────────────────────────
        adj_bonus, adj_pairs = self._compute_adjacency(expected_set, broad_hits)

        total_penalty = ori_pen + amr_pen + unreq_pen + len_pen
        total_bonus = adj_bonus
        multiplier = 1.0 + total_bonus - total_penalty

        return {
            "structural_multiplier": round(multiplier, 4),
            "ori_count": n_ori,
            "ori_penalty": round(ori_pen, 4),
            "ori_tokens": sorted(ori_found),
            "ori_loci": [sorted(locus["tokens"]) for locus in ori_loci],
            "amr_count": n_amr,
            "amr_penalty": round(amr_pen, 4),
            "unrequested_tokens": unrequested,
            "unrequested_penalty": round(unreq_pen, 4),
            "seq_len": seq_len,
            "length_penalty": round(len_pen, 4),
            "adjacency_bonus": round(adj_bonus, 4),
            "adjacency_pairs": adj_pairs,
            "total_penalty": round(total_penalty, 4),
            "total_bonus": round(total_bonus, 4),
            "found_tokens": sorted(found_tokens),
        }

    def _compute_adjacency(
        self,
        expected_set: set[str],
        broad_hits: dict[str, dict],
    ) -> tuple[float, list[tuple[str, str]]]:
        """Check promoter–CDS adjacency for requested pairs."""
        bonus = 0.0
        pairs: list[tuple[str, str]] = []

        prom_tokens = {t for t in expected_set if _extract_category(t) == "PROM"}
        cds_tokens = {
            t for t in expected_set
            if _extract_category(t) in CDS_CATEGORIES
        }

        for prom in prom_tokens:
            if prom not in broad_hits:
                continue

            affinity = PROMOTER_CDS_AFFINITY.get(prom)
            if affinity is not None and len(affinity) == 0:
                continue  # e.g. U6 — drives gRNAs, skip

            # which CDS tokens does this promoter care about?
            if affinity is None:
                candidates = cds_tokens
            else:
                candidates = cds_tokens & affinity

            prom_hit = broad_hits[prom]
            prom_lo = min(prom_hit["qstart"], prom_hit["qend"])
            prom_hi = max(prom_hit["qstart"], prom_hit["qend"])

            for cds in candidates:
                if cds not in broad_hits:
                    continue
                cds_hit = broad_hits[cds]
                cds_lo = min(cds_hit["qstart"], cds_hit["qend"])
                cds_hi = max(cds_hit["qstart"], cds_hit["qend"])

                # min gap between the two regions
                if prom_hi <= cds_lo:
                    gap = cds_lo - prom_hi
                elif cds_hi <= prom_lo:
                    gap = prom_lo - cds_hi
                else:
                    gap = 0  # overlapping

                if gap <= self.adjacency_max_gap:
                    bonus += self.adjacency_bonus_weight
                    pairs.append((prom, cds))

        return bonus, pairs

    def _compute_length_penalty(self, seq_len: int) -> float:
        excess_kb = max(0.0, (seq_len - self.length_soft_max) / 1000.0)
        return min(excess_kb * self.length_penalty_per_kb, self.length_penalty_cap)

    # ------------------------------------------------ Scorer interface
    def score_sequence(self, prompt: str, sequence: str, **kwargs) -> float:
        return self.score_sequence_detailed(prompt, sequence, **kwargs)["reward"]

    def score_sequence_detailed(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> dict[str, Any]:
        seq, _ = _clean_dna(sequence)
        expected_tokens = _parse_hard_tokens(prompt)

        if not expected_tokens or len(seq) < 20:
            return {
                "reward": 0.0,
                "base_reward": 0.0,
                "structural_multiplier": 1.0,
                "expected": len(expected_tokens),
                "found": 0,
                "token_scores": {},
                "penalties": {},
            }

        # 1. Base plannotate composite
        base_result = self._base.score_sequence_detailed(prompt, sequence, **kwargs)
        base_reward = base_result["reward"]

        # 2. Broad BLAST for all elements
        broad = self._run_broad_blast([("query", seq)])
        broad_hits = broad.get("query", {})

        # 3. Structural penalties
        struct = self._compute_structural(expected_tokens, broad_hits, len(seq))
        multiplier = struct["structural_multiplier"]

        final = max(0.0, min(1.0, base_reward * multiplier))

        return {
            "reward": round(final, 4),
            "base_reward": round(base_reward, 4),
            **struct,
            "base_detail": base_result,
        }

    def score_batch(
        self,
        prompts: list[str],
        sequences: list[str],
        **kwargs,
    ) -> list[float]:
        """Batch-optimised: two BLAST calls (base + broad) for all sequences."""
        # Prepare queries
        all_expected: list[list[str]] = []
        query_seqs: list[tuple[str, str]] = []
        clean_seqs: list[str] = []
        all_sseqid_to_token: dict[str, str] = {}

        for i, (prompt, sequence) in enumerate(zip(prompts, sequences)):
            seq, _ = _clean_dna(sequence)
            clean_seqs.append(seq)
            expected = _parse_hard_tokens(prompt)
            all_expected.append(expected)

            if not expected or len(seq) < 20:
                continue

            _, s2t = self._base._resolve_tokens(expected)
            all_sseqid_to_token.update(s2t)
            query_seqs.append((f"q{i}", seq))

        # Base BLAST (expected tokens only, for plannotate composite)
        base_hits = self._base._run_blast_batch(query_seqs, all_sseqid_to_token)

        # Broad BLAST (all tokens, for structural penalties)
        broad_hits = self._run_broad_blast(query_seqs)

        rewards: list[float] = []
        for i, expected in enumerate(all_expected):
            if not expected or len(clean_seqs[i]) < 20:
                rewards.append(0.0)
                continue

            qid = f"q{i}"

            # base composite
            per_token = base_hits.get(qid, {})
            base_result = PlannotateScorer._compute_composite(expected, per_token)
            base_reward = base_result["reward"]

            # structural
            broad = broad_hits.get(qid, {})
            struct = self._compute_structural(expected, broad, len(clean_seqs[i]))
            multiplier = struct["structural_multiplier"]

            final = max(0.0, min(1.0, base_reward * multiplier))
            rewards.append(round(final, 4))

        return rewards
