"""Plannotate BLAST scorer.

Uses a local BLAST database built from the plannotate feature database
to score generated plasmid sequences.  For each expected annotation token
in the prompt, we look up matching reference sequences (via the motif
registry bridge) and BLAST the candidate against them.

The motif registry provides the mapping from annotation tokens
(e.g. <AMR_AMPICILLIN>) to gene-level sseqid names (e.g. AmpR, AmpR_(2))
used in the plannotate DB.  Without this bridge, naive substring matching
fails because tokens use drug/feature names while plannotate uses gene names.

Performance notes
-----------------
* The BLAST DB is filtered to only include sequences that are actually
  referenced by the token bridge (~600 vs 567 K).  This gives a ~900x
  speedup in BLAST search time.
* ``score_batch`` writes all queries to a single multi-record FASTA and
  runs BLAST once, avoiding per-sequence subprocess overhead.
* Tabular output (``-outfmt 6``) is used instead of XML for faster parsing.

Ray integration
---------------
The scorer holds on-disk BLAST databases and is **not** designed to be
pickled and shipped across workers.  Instead, wrap it in a Ray actor::

    @ray.remote
    class ScorerActor:
        def __init__(self, plannotate_db_path, motif_registry_path, db_dir):
            self.scorer = PlannotateScorer(
                plannotate_db_path=plannotate_db_path,
                motif_registry_path=motif_registry_path,
                db_dir=db_dir,
            )

        def score_batch(self, prompts, sequences):
            return self.scorer.score_batch(prompts, sequences)

Each actor builds its own BLAST DB on its local filesystem once.  Use an
``ActorPool`` for concurrency.

Requires NCBI BLAST+ command-line tools (makeblastdb, blastn, blastx).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from typing import Any

import pandas as pd

from post_training.scorers.base import Scorer

HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}

_BLASTN_COLS = (
    "qseqid sseqid pident length mismatch gapopen "
    "qstart qend sstart send evalue bitscore nident slen"
)
_BLASTX_COLS = _BLASTN_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hard_tokens(prompt: str) -> list[str]:
    """Extract hard annotation tokens from a prompt string."""
    all_tokens = re.findall(r"<[^>]+>", prompt)
    return [
        t for t in all_tokens
        if any(t.strip("<>").startswith(p) for p in HARD_PREFIXES)
    ]


def _clean_dna(text: str) -> tuple[str, bool]:
    """Strip special tokens, return (clean_dna, has_eos)."""
    raw = text.upper()
    has_eos = "<EOS>" in text or "</s>" in text
    seq = re.sub(r"<[^>]+>", "", raw)
    seq = re.sub(r"[^ATGCN]", "", seq)
    return seq, has_eos


def _sanitize_id(raw: str) -> str:
    """Make an ID safe for BLAST FASTA headers."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def _build_blast_db(
    sequences: list[tuple[str, str]],
    db_path: str,
    dbtype: str = "nucl",
) -> str:
    """Write sequences to FASTA and build a BLAST database.

    Deduplicates by appending an index for repeated cleaned IDs.
    """
    fasta_path = db_path + ".fasta"
    seen: dict[str, int] = {}
    with open(fasta_path, "w") as f:
        for seq_id, seq in sequences:
            cid = _sanitize_id(seq_id)
            n = seen.get(cid, 0)
            seen[cid] = n + 1
            unique_id = cid if n == 0 else f"{cid}__dup{n}"
            f.write(f">{unique_id}\n{seq}\n")

    cmd = [
        "makeblastdb",
        "-in", fasta_path,
        "-dbtype", dbtype,
        "-out", db_path,
        "-parse_seqids",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return db_path


def _build_token_bridge(
    motif_registry: pd.DataFrame,
    plannotate_db: pd.DataFrame,
) -> dict[str, list[str]]:
    """Build mapping:  token -> list of plannotate sseqids.

    Uses exact sseqid matching (case-insensitive fallback) between the
    motif registry and plannotate DB.
    """
    plannotate_sseqids = set(plannotate_db["sseqid"].unique())
    plannotate_lower = {s.lower(): s for s in plannotate_sseqids}

    bridge: dict[str, list[str]] = {}
    for token in motif_registry["token"].unique():
        motif_rows = motif_registry[motif_registry["token"] == token]
        matched = []
        for sseqid in motif_rows["sseqid"].unique():
            s = str(sseqid)
            if s in plannotate_sseqids:
                matched.append(s)
            elif s.lower() in plannotate_lower:
                matched.append(plannotate_lower[s.lower()])
        if matched:
            bridge[token] = matched
    return bridge


def _parse_tabular_hits(
    tsv_path: str,
    sseqid_to_token: dict[str, str],
    fasta_id_to_sseqid: dict[str, str],
) -> dict[str, dict[str, dict]]:
    """Parse BLAST tabular output into per-query, per-token best hits.

    Returns ``{query_id: {token: hit_info}}``
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
            evalue = float(parts[10])
            bitscore = float(parts[11])
            nident = int(parts[12])
            slen = int(parts[13])

            token = _resolve_sseqid(raw_sseqid, sseqid_to_token, fasta_id_to_sseqid)
            if token is None:
                continue

            coverage = min((aln_len / max(slen, 1)) * 100, 100.0)
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
            }

            per_token = results.setdefault(qseqid, {})
            prev = per_token.get(token)
            if prev is None or hit_info["bit_score"] > prev["bit_score"]:
                per_token[token] = hit_info

    return results


def _resolve_sseqid(
    raw_sseqid: str,
    sseqid_to_token: dict[str, str],
    fasta_id_to_sseqid: dict[str, str],
) -> str | None:
    """Map a BLAST subject ID back to a token."""
    cid = raw_sseqid.rsplit("|", 1)[-1] if "|" in raw_sseqid else raw_sseqid

    if cid in sseqid_to_token:
        return sseqid_to_token[cid]

    orig = fasta_id_to_sseqid.get(cid)
    if orig and orig in sseqid_to_token:
        return sseqid_to_token[orig]

    return None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PlannotateScorer(Scorer):
    """BLAST-based scorer using the plannotate feature database.

    Uses the motif registry as a bridge to correctly map annotation tokens
    to plannotate reference sequences, then BLASTs the candidate plasmid
    against those references.

    Composite score::

        quality    = geo_mean(found_token_scores) ^ sharpness
        recall_pen = recall_floor + (1 - recall_floor) * recall
        composite  = quality * recall_pen
    """

    def __init__(
        self,
        plannotate_db_path: str | None = None,
        plannotate_df: pd.DataFrame | None = None,
        motif_registry_path: str | None = None,
        motif_registry_df: pd.DataFrame | None = None,
        db_dir: str | None = None,
        evalue: float = 1e-5,
        filter_db: bool = True,
    ):
        if plannotate_df is not None:
            plannotate_full = plannotate_df
        elif plannotate_db_path is not None:
            plannotate_full = pd.read_parquet(plannotate_db_path)
        else:
            raise ValueError("Must provide plannotate_db_path or plannotate_df")

        if motif_registry_df is not None:
            self.motif_registry = motif_registry_df
        elif motif_registry_path is not None:
            self.motif_registry = pd.read_parquet(motif_registry_path)
        else:
            raise ValueError("Must provide motif_registry_path or motif_registry_df")

        self.evalue = evalue
        self._token_bridge = _build_token_bridge(
            self.motif_registry, plannotate_full
        )

        if filter_db:
            bridge_sseqids = set()
            for sids in self._token_bridge.values():
                bridge_sseqids.update(sids)
            self.plannotate_df = plannotate_full[
                plannotate_full["sseqid"].isin(bridge_sseqids)
            ].copy()
        else:
            self.plannotate_df = plannotate_full

        if db_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="plannotate_blast_")
            db_dir = self._tmp_dir
        else:
            self._tmp_dir = None
            os.makedirs(db_dir, exist_ok=True)

        self.db_dir = db_dir
        self._nucl_db: str | None = None
        self._prot_db: str | None = None
        self._fasta_id_to_sseqid: dict[str, str] = {}

        self._build_databases()

    # ------------------------------------------------------------------ DB
    def _build_databases(self):
        """Build nucleotide and protein BLAST databases."""
        nucl_seqs: list[tuple[str, str]] = []
        prot_seqs: list[tuple[str, str]] = []
        seen: dict[str, int] = {}

        for _, row in self.plannotate_df.iterrows():
            sseqid = str(row["sseqid"])
            seq = str(row["sequence"]).strip()
            if not seq or len(seq) < 10:
                continue

            cid = _sanitize_id(sseqid)
            n = seen.get(cid, 0)
            seen[cid] = n + 1
            unique_id = cid if n == 0 else f"{cid}__dup{n}"
            self._fasta_id_to_sseqid[unique_id] = sseqid

            if row["seq_type"] == "nucleotide":
                nucl_seqs.append((sseqid, seq))
            elif row["seq_type"] == "protein":
                prot_seqs.append((sseqid, seq))

        if nucl_seqs:
            self._nucl_db = _build_blast_db(
                nucl_seqs,
                os.path.join(self.db_dir, "plannotate_nucl"),
                dbtype="nucl",
            )

        if prot_seqs:
            self._prot_db = _build_blast_db(
                prot_seqs,
                os.path.join(self.db_dir, "plannotate_prot"),
                dbtype="prot",
            )

    # -------------------------------------------------------------- tokens
    def _resolve_tokens(
        self, tokens: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Map tokens -> plannotate sseqids using the bridge."""
        token_to_sseqids: dict[str, list[str]] = {}
        sseqid_to_token: dict[str, str] = {}
        for token in tokens:
            sseqids = self._token_bridge.get(token, [])
            if sseqids:
                token_to_sseqids[token] = sseqids
                for sid in sseqids:
                    sseqid_to_token[_sanitize_id(sid)] = token
                    sseqid_to_token[sid] = token
        return token_to_sseqids, sseqid_to_token

    # --------------------------------------------------------------- BLAST
    def _run_blast_batch(
        self,
        query_seqs: list[tuple[str, str]],
        sseqid_to_token: dict[str, str],
    ) -> dict[str, dict[str, dict]]:
        """Run BLAST for multiple queries in a single subprocess call.

        Args:
            query_seqs: list of (query_id, dna_sequence) pairs
            sseqid_to_token: reverse lookup from sseqid -> token

        Returns:
            ``{query_id: {token: hit_info}}``
        """
        if not query_seqs:
            return {}

        query_fasta = os.path.join(self.db_dir, "query_batch.fasta")
        with open(query_fasta, "w") as f:
            for qid, seq in query_seqs:
                f.write(f">{qid}\n{seq}\n")

        merged: dict[str, dict[str, dict]] = {}

        if self._nucl_db:
            out_tsv = os.path.join(self.db_dir, "blastn_batch.tsv")
            self._run_blast_cmd(
                "blastn", query_fasta, self._nucl_db, out_tsv,
                evalue_override=10.0,
                extra_args=["-word_size", "7"],
            )
            hits = _parse_tabular_hits(
                out_tsv, sseqid_to_token, self._fasta_id_to_sseqid,
            )
            for qid, per_token in hits.items():
                dest = merged.setdefault(qid, {})
                for tok, info in per_token.items():
                    prev = dest.get(tok)
                    if prev is None or info["bit_score"] > prev["bit_score"]:
                        dest[tok] = info

        if self._prot_db:
            out_tsv = os.path.join(self.db_dir, "blastx_batch.tsv")
            self._run_blast_cmd(
                "blastx", query_fasta, self._prot_db, out_tsv,
            )
            hits = _parse_tabular_hits(
                out_tsv, sseqid_to_token, self._fasta_id_to_sseqid,
            )
            for qid, per_token in hits.items():
                dest = merged.setdefault(qid, {})
                for tok, info in per_token.items():
                    prev = dest.get(tok)
                    if prev is None or info["bit_score"] > prev["bit_score"]:
                        dest[tok] = info

        return merged

    def _run_blast_cmd(
        self,
        program: str,
        query_fasta: str,
        db_path: str,
        out_path: str,
        evalue_override: float | None = None,
        extra_args: list[str] | None = None,
    ):
        effective_evalue = evalue_override if evalue_override is not None else self.evalue
        cols = _BLASTN_COLS if program == "blastn" else _BLASTX_COLS
        cmd = [
            program,
            "-query", query_fasta,
            "-db", db_path,
            "-out", out_path,
            "-outfmt", f"6 {cols}",
            "-evalue", str(effective_evalue),
            "-max_target_seqs", "500",
            "-num_threads", "4",
        ]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, capture_output=True, text=True)

    # --------------------------------------------------- single-query BLAST
    def _run_blast(
        self,
        query_seq: str,
        tokens: list[str],
    ) -> dict[str, dict]:
        """Run BLAST for a single query sequence."""
        _, sseqid_to_token = self._resolve_tokens(tokens)
        if not sseqid_to_token:
            return {}

        results = self._run_blast_batch(
            [("query", query_seq)], sseqid_to_token,
        )
        return results.get("query", {})

    # ----------------------------------------------------------- composite
    @staticmethod
    def _compute_composite(
        expected_tokens: list[str],
        best_per_token: dict[str, dict],
        w_id: float = 0.4,
        w_cov: float = 0.35,
        w_norm: float = 0.25,
        norm_score_cap: float = 5.0,
        sharpness: float = 2.0,
        recall_floor: float = 0.5,
    ) -> dict[str, Any]:
        token_scores = {}
        found_scores = []

        for tok in expected_tokens:
            if tok in best_per_token:
                h = best_per_token[tok]
                id_score = min(h["pct_id"] / 100.0, 1.0)
                cov_score = min(h["coverage"] / 100.0, 1.0)
                norm_scaled = min(h["norm_score"] / norm_score_cap, 1.0)
                quality = w_id * id_score + w_cov * cov_score + w_norm * norm_scaled

                token_scores[tok] = {
                    "score": round(quality, 4),
                    "found": True,
                    "pct_id": h["pct_id"],
                    "coverage": h["coverage"],
                    "norm_score": h["norm_score"],
                    "bit_score": h["bit_score"],
                    "evalue": h["evalue"],
                    "sseqid": h["sseqid"],
                }
                found_scores.append(quality)
            else:
                token_scores[tok] = {
                    "score": 0.0, "found": False,
                    "pct_id": 0, "coverage": 0, "norm_score": 0,
                    "bit_score": 0, "evalue": None, "sseqid": None,
                }

        n_expected = len(expected_tokens)
        n_found = len(found_scores)
        recall = n_found / n_expected if n_expected > 0 else 0.0

        if found_scores:
            log_scores = [math.log(max(s, 1e-6)) for s in found_scores]
            geo_mean = math.exp(sum(log_scores) / len(log_scores))
            quality = geo_mean ** sharpness
        else:
            geo_mean = 0.0
            quality = 0.0

        recall_penalty = recall_floor + (1.0 - recall_floor) * recall
        composite = quality * recall_penalty

        return {
            "reward": round(composite, 4),
            "composite": round(composite, 4),
            "quality": round(quality, 4),
            "geo_mean": round(geo_mean, 4),
            "recall": round(recall, 4),
            "recall_penalty": round(recall_penalty, 4),
            "found": n_found,
            "expected": n_expected,
            "token_scores": token_scores,
            "hits": [best_per_token[t] for t in expected_tokens if t in best_per_token],
        }

    # ------------------------------------------------ Scorer interface
    def score_sequence(
        self,
        prompt: str,
        sequence: str,
        w_id: float = 0.4,
        w_cov: float = 0.35,
        w_norm: float = 0.25,
        norm_score_cap: float = 5.0,
        sharpness: float = 2.0,
        recall_floor: float = 0.5,
        **kwargs,
    ) -> float:
        return self.score_sequence_detailed(
            prompt, sequence,
            w_id=w_id, w_cov=w_cov, w_norm=w_norm,
            norm_score_cap=norm_score_cap,
            sharpness=sharpness, recall_floor=recall_floor,
            **kwargs,
        )["reward"]

    def score_sequence_detailed(
        self,
        prompt: str,
        sequence: str,
        w_id: float = 0.4,
        w_cov: float = 0.35,
        w_norm: float = 0.25,
        norm_score_cap: float = 5.0,
        sharpness: float = 2.0,
        recall_floor: float = 0.5,
        **kwargs,
    ) -> dict[str, Any]:
        seq, _ = _clean_dna(sequence)
        expected_tokens = _parse_hard_tokens(prompt)

        if not expected_tokens or len(seq) < 20:
            return {
                "reward": 0.0, "composite": 0.0, "quality": 0.0,
                "recall": 0.0, "found": 0,
                "expected": len(expected_tokens),
                "token_scores": {}, "hits": [],
            }

        best_per_token = self._run_blast(seq, expected_tokens)
        return self._compute_composite(
            expected_tokens, best_per_token,
            w_id=w_id, w_cov=w_cov, w_norm=w_norm,
            norm_score_cap=norm_score_cap,
            sharpness=sharpness, recall_floor=recall_floor,
        )

    def score_batch(
        self,
        prompts: list[str],
        sequences: list[str],
        w_id: float = 0.4,
        w_cov: float = 0.35,
        w_norm: float = 0.25,
        norm_score_cap: float = 5.0,
        sharpness: float = 2.0,
        recall_floor: float = 0.5,
        **kwargs,
    ) -> list[float]:
        """Batch-optimised scoring: single BLAST call for all sequences."""
        query_seqs = []
        all_tokens: list[list[str]] = []
        all_sseqid_to_token: dict[str, str] = {}

        for i, (prompt, sequence) in enumerate(zip(prompts, sequences)):
            seq, _ = _clean_dna(sequence)
            expected = _parse_hard_tokens(prompt)
            all_tokens.append(expected)

            if not expected or len(seq) < 20:
                continue

            _, s2t = self._resolve_tokens(expected)
            all_sseqid_to_token.update(s2t)
            query_seqs.append((f"q{i}", seq))

        batch_hits = self._run_blast_batch(query_seqs, all_sseqid_to_token)

        rewards = []
        for i, expected in enumerate(all_tokens):
            if not expected:
                rewards.append(0.0)
                continue
            per_token = batch_hits.get(f"q{i}", {})
            result = self._compute_composite(
                expected, per_token,
                w_id=w_id, w_cov=w_cov, w_norm=w_norm,
                norm_score_cap=norm_score_cap,
                sharpness=sharpness, recall_floor=recall_floor,
            )
            rewards.append(result["reward"])

        return rewards
