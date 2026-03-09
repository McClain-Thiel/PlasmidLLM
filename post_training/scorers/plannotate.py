"""Plannotate BLAST scorer.

Uses a local BLAST database built from the plannotate feature database
to score generated plasmid sequences. For each expected annotation token
in the prompt, we look up matching reference sequences and BLAST the
candidate against them. The composite score captures both the quality
of individual hits and the fraction of expected features recovered.

Requires NCBI BLAST+ command-line tools (makeblastdb, blastn, tblastn).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Any

import pandas as pd
from Bio import Blast

from post_training.scorers.base import Scorer

HARD_PREFIXES = {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"}


def _parse_hard_tokens(prompt: str) -> list[str]:
    """Extract hard annotation tokens from a prompt string."""
    all_tokens = re.findall(r"<[^>]+>", prompt)
    hard = []
    for t in all_tokens:
        inner = t.strip("<>")
        if any(inner.startswith(p) for p in HARD_PREFIXES):
            hard.append(t)
    return hard


def _clean_dna(text: str) -> tuple[str, bool]:
    """Strip special tokens, return (clean_dna, has_eos)."""
    raw = text.upper()
    has_eos = "<EOS>" in text or "</s>" in text
    seq = re.sub(r"<[^>]+>", "", raw)
    seq = re.sub(r"[^ATGCN]", "", seq)
    return seq, has_eos


def _build_blast_db(
    sequences: list[tuple[str, str]],
    db_path: str,
    dbtype: str = "nucl",
) -> str:
    """Write sequences to FASTA and build a BLAST database.

    Args:
        sequences: list of (id, sequence) pairs
        db_path: path prefix for the database files
        dbtype: 'nucl' or 'prot'

    Returns:
        The db_path for use in blast commands.
    """
    fasta_path = db_path + ".fasta"
    with open(fasta_path, "w") as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")

    cmd = [
        "makeblastdb",
        "-in", fasta_path,
        "-dbtype", dbtype,
        "-out", db_path,
        "-parse_seqids",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return db_path


class PlannotateScorer(Scorer):
    """BLAST-based scorer using the plannotate feature database.

    Builds local BLAST databases from the plannotate parquet and scores
    candidate plasmid sequences by BLASTing them against expected features.

    Composite score:
        quality    = geo_mean(found_token_scores) ^ sharpness
        recall_pen = recall_floor + (1 - recall_floor) * recall
        composite  = quality * recall_pen
    """

    def __init__(
        self,
        plannotate_db_path: str | None = None,
        plannotate_df: pd.DataFrame | None = None,
        db_dir: str | None = None,
        evalue: float = 1e-5,
    ):
        if plannotate_df is not None:
            self.df = plannotate_df
        elif plannotate_db_path is not None:
            self.df = pd.read_parquet(plannotate_db_path)
        else:
            raise ValueError("Must provide plannotate_db_path or plannotate_df")

        self.evalue = evalue

        if db_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="plannotate_blast_")
            db_dir = self._tmp_dir
        else:
            self._tmp_dir = None
            os.makedirs(db_dir, exist_ok=True)

        self.db_dir = db_dir

        self._token_to_sseqids: dict[str, list[str]] = {}
        self._sseqid_to_token: dict[str, str] = {}

        self._nucl_db: str | None = None
        self._prot_db: str | None = None

        self._build_databases()

    def _build_databases(self):
        """Build nucleotide and protein BLAST databases from the dataframe."""
        nucl_seqs = []
        prot_seqs = []

        for _, row in self.df.iterrows():
            sseqid = str(row["sseqid"])
            seq = str(row["sequence"]).strip()
            if not seq or len(seq) < 10:
                continue

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

    def _build_token_index(self, tokens: list[str]):
        """Build mapping from tokens to sseqids in the dataframe.

        The plannotate db doesn't have a 'token' column, but the motif registry
        does. We need to map from our annotation tokens to plannotate sseqids.
        Since the plannotate db uses descriptive sseqid names rather than
        angle-bracket tokens, we do a fuzzy join via the `features` column.
        """
        self._token_to_sseqids.clear()
        self._sseqid_to_token.clear()

        for token in tokens:
            inner = token.strip("<>")
            parts = inner.split("_", 1)
            if len(parts) < 2:
                continue
            feature_name = parts[1].lower()

            mask = self.df["features"].str.lower().str.contains(
                feature_name, na=False, regex=False
            ) | self.df["sseqid"].str.lower().str.contains(
                feature_name, na=False, regex=False
            )
            matching = self.df[mask]

            if len(matching) > 0:
                sseqids = matching["sseqid"].unique().tolist()
                self._token_to_sseqids[token] = sseqids
                for sid in sseqids:
                    self._sseqid_to_token[str(sid)] = token

    def _run_blast(
        self,
        query_seq: str,
        tokens: list[str],
    ) -> dict[str, dict]:
        """Run BLAST searches and return best hit per token."""
        self._build_token_index(tokens)

        all_target_sseqids = set()
        for sids in self._token_to_sseqids.values():
            all_target_sseqids.update(sids)

        if not all_target_sseqids:
            return {}

        query_fasta = os.path.join(self.db_dir, "query.fasta")
        with open(query_fasta, "w") as f:
            f.write(f">query\n{query_seq}\n")

        best_per_token: dict[str, dict] = {}

        if self._nucl_db:
            self._blast_search(
                query_fasta, self._nucl_db, "blastn",
                all_target_sseqids, best_per_token,
            )

        if self._prot_db:
            self._blast_search(
                query_fasta, self._prot_db, "tblastn",
                all_target_sseqids, best_per_token,
            )

        return best_per_token

    def _blast_search(
        self,
        query_fasta: str,
        db_path: str,
        program: str,
        target_sseqids: set[str],
        best_per_token: dict[str, dict],
    ):
        """Execute a BLAST search and update best_per_token."""
        out_xml = os.path.join(self.db_dir, f"{program}_results.xml")

        cmd = [
            program,
            "-query", query_fasta,
            "-db", db_path,
            "-out", out_xml,
            "-outfmt", "5",
            "-evalue", str(self.evalue),
            "-max_target_seqs", "500",
            "-num_threads", "4",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return

        try:
            blast_record = Blast.read(out_xml)
        except Exception:
            return

        for hit in blast_record:
            if not hit:
                continue

            target_id = hit.target.id
            clean_id = target_id.split("|")[-1] if "|" in target_id else target_id

            if clean_id not in self._sseqid_to_token:
                for sid in target_sseqids:
                    if sid in target_id or target_id in sid:
                        clean_id = sid
                        break

            if clean_id not in self._sseqid_to_token:
                continue

            token = self._sseqid_to_token[clean_id]

            for alignment in hit:
                score = alignment.score
                annotations = alignment.annotations

                evalue = annotations.get("evalue", float("inf"))
                bit_score = annotations.get("bit score", 0)
                identity = annotations.get("identity", 0)

                target_len = len(hit.target.seq) if hit.target.seq else 1
                pct_id = (identity / max(target_len, 1)) * 100
                coverage = (identity / max(target_len, 1)) * 100

                norm_score = bit_score / max(target_len, 1)

                hit_info = {
                    "token": token,
                    "sseqid": clean_id,
                    "score": score,
                    "bit_score": bit_score,
                    "evalue": evalue,
                    "pct_id": round(pct_id, 2),
                    "coverage": round(coverage, 2),
                    "norm_score": round(norm_score, 4),
                    "identity": identity,
                    "target_len": target_len,
                }

                prev = best_per_token.get(token)
                if prev is None or hit_info["bit_score"] > prev["bit_score"]:
                    best_per_token[token] = hit_info

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
        result = self.score_sequence_detailed(
            prompt, sequence,
            w_id=w_id, w_cov=w_cov, w_norm=w_norm,
            norm_score_cap=norm_score_cap,
            sharpness=sharpness,
            recall_floor=recall_floor,
            **kwargs,
        )
        return result["reward"]

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
        seq, has_eos = _clean_dna(sequence)
        expected_tokens = _parse_hard_tokens(prompt)

        if not expected_tokens or len(seq) < 20:
            return {
                "reward": 0.0,
                "composite": 0.0,
                "quality": 0.0,
                "recall": 0.0,
                "found": 0,
                "expected": len(expected_tokens),
                "token_scores": {},
                "hits": [],
            }

        best_per_token = self._run_blast(seq, expected_tokens)

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
                    "score": 0.0,
                    "found": False,
                    "pct_id": 0,
                    "coverage": 0,
                    "norm_score": 0,
                    "bit_score": 0,
                    "evalue": None,
                    "sseqid": None,
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
            "hits": list(best_per_token.values()),
        }
