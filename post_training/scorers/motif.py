"""CIGAR-based motif scorer using parasail semi-global alignment.

Uses sg_qx (semi-global, free query end gaps) alignment with full CIGAR
traceback to compute percent identity, coverage, and normalized score.
Supports both DNA and protein (6-frame translation) alignment.

Three-pass approach for speed:
  1. K-mer pre-filter: skip motifs with low k-mer overlap with target
  2. Score-only alignment: skip motifs with low normalized score
  3. Full trace alignment: only for candidates passing both filters

This scorer was developed in pretrained_analysis.ipynb and provides
richer alignment detail than the simpler score-ratio AlignmentScorer.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import parasail
from Bio.Seq import Seq

from post_training.scorers.base import Scorer


class MotifScorer(Scorer):
    """Alignment-based scorer using CIGAR parsing for detailed motif matching.

    Computes a composite score:
        quality    = geo_mean(found_token_scores) ^ sharpness
        recall_pen = recall_floor + (1 - recall_floor) * recall
        composite  = quality * recall_pen
    """

    def __init__(
        self,
        motif_db=None,
        motif_db_path: str | None = None,
        dna_gap_open: int = 5,
        dna_gap_extend: int = 2,
        prot_gap_open: int = 11,
        prot_gap_extend: int = 1,
    ):
        """Initialize scorer.

        Args:
            motif_db: List of dicts or DataFrame with columns: token, sequence, seq_type.
            motif_db_path: Path to parquet file (alternative to motif_db).
            dna_gap_open: Gap open penalty for DNA alignment.
            dna_gap_extend: Gap extend penalty for DNA alignment.
            prot_gap_open: Gap open penalty for protein alignment.
            prot_gap_extend: Gap extend penalty for protein alignment.
        """
        self.dna_gap_open = dna_gap_open
        self.dna_gap_extend = dna_gap_extend
        self.prot_gap_open = prot_gap_open
        self.prot_gap_extend = prot_gap_extend

        if motif_db is not None:
            self.motif_db = self._normalize_db(motif_db)
        elif motif_db_path is not None:
            import pandas as pd
            df = pd.read_parquet(motif_db_path)
            self.motif_db = df.to_dict("records")
        else:
            raise ValueError("Must provide either motif_db or motif_db_path")

    @staticmethod
    def _normalize_db(motif_db) -> list[dict]:
        """Convert DataFrame to list of dicts if needed."""
        try:
            import pandas as pd
            if isinstance(motif_db, pd.DataFrame):
                return motif_db.to_dict("records")
        except ImportError:
            pass
        return motif_db

    # ── Sequence utilities ────────────────────────────────────────────────

    @staticmethod
    def _clean_seq(seq):
        lines = seq.strip().split("\n")
        lines = [l for l in lines if not l.startswith(">")]
        cleaned = "".join(lines).upper()
        cleaned = re.sub(r"[^ATGCNRYSWKMBDHV]", "", cleaned)
        return cleaned

    @staticmethod
    def _kmer_set(seq, k=15):
        """Extract set of k-mers from a sequence for fast overlap checking."""
        if len(seq) < k:
            return set()
        return {seq[i : i + k] for i in range(len(seq) - k + 1)}

    def _get_six_frames(self, dna):
        s = Seq(dna)
        frames = []
        for i in range(3):
            f_seq = s[i:]
            f_seq = f_seq[: len(f_seq) - (len(f_seq) % 3)]
            frames.append(str(f_seq.translate(stop_symbol="X")))

            r_seq = s.reverse_complement()[i:]
            r_seq = r_seq[: len(r_seq) - (len(r_seq) % 3)]
            frames.append(str(r_seq.translate(stop_symbol="X")))
        return frames

    # ── CIGAR parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_cigar(cigar_str):
        """Parse CIGAR string, stripping leading/trailing insertions."""
        ops = re.findall(r"(\d+)([MIDX=])", cigar_str)

        while ops and ops[0][1] == "I":
            ops.pop(0)
        while ops and ops[-1][1] == "I":
            ops.pop()

        all_ops = re.findall(r"(\d+)([MIDX=])", cigar_str)
        leading_i = 0
        for length_str, op in all_ops:
            if op == "I":
                leading_i += int(length_str)
            else:
                break

        matches = 0
        mismatches = 0
        ins = 0
        dels = 0
        query_pos = leading_i
        ref_pos = 0
        start_query = None
        end_query = None

        for length_str, op in ops:
            length = int(length_str)
            if op in ("=", "M"):
                if start_query is None:
                    start_query = query_pos
                end_query = query_pos + length - 1
                matches += length
                query_pos += length
                ref_pos += length
            elif op == "X":
                if start_query is None:
                    start_query = query_pos
                end_query = query_pos + length - 1
                mismatches += length
                query_pos += length
                ref_pos += length
            elif op == "I":
                ins += length
                query_pos += length
            elif op == "D":
                dels += length
                ref_pos += length

        return {
            "matches": matches,
            "mismatches": mismatches,
            "ins": ins,
            "dels": dels,
            "core_len": matches + mismatches,
            "start_query": start_query if start_query is not None else 0,
            "end_query": end_query if end_query is not None else 0,
            "ref_consumed": ref_pos,
        }

    # ── Alignment ─────────────────────────────────────────────────────────

    def _align_dna_score_only(self, target_dna, motif_seq):
        """Fast score-only alignment — no traceback, no CIGAR."""
        res = parasail.sg_qx_striped_sat(
            target_dna, motif_seq,
            self.dna_gap_open, self.dna_gap_extend,
            parasail.dnafull,
        )
        return res.score / len(motif_seq) if len(motif_seq) > 0 else 0.0

    def _align_dna(self, target_dna, motif_seq):
        res = parasail.sg_qx_trace_striped_sat(
            target_dna, motif_seq,
            self.dna_gap_open, self.dna_gap_extend,
            parasail.dnafull,
        )
        cigar_str = res.cigar.decode.decode()
        parsed = self._parse_cigar(cigar_str)
        return self._build_hit(res.score, parsed, len(motif_seq), seq_type="dna")

    def _align_protein(self, target_dna, protein_frames, motif_seq, target_len):
        best_hit = None

        for idx, frame_seq in enumerate(protein_frames):
            if not frame_seq:
                continue
            res = parasail.sg_qx_trace_striped_sat(
                frame_seq, motif_seq,
                self.prot_gap_open, self.prot_gap_extend,
                parasail.blosum62,
            )
            cigar_str = res.cigar.decode.decode()
            parsed = self._parse_cigar(cigar_str)
            hit = self._build_hit(
                res.score, parsed, len(motif_seq),
                seq_type="protein", frame=idx,
            )

            # Map protein coords → DNA coords for cross-type dedup
            is_reverse = idx % 2 == 1
            frame_offset = idx // 2
            dna_start = hit["start_pos"] * 3 + frame_offset
            dna_end = (hit["end_pos"] + 1) * 3 - 1 + frame_offset
            if is_reverse:
                dna_start_tmp = target_len - 1 - dna_end
                dna_end = target_len - 1 - (hit["start_pos"] * 3 + frame_offset)
                dna_start = dna_start_tmp
            hit["dna_start"] = dna_start
            hit["dna_end"] = dna_end

            if best_hit is None or hit["score"] > best_hit["score"]:
                best_hit = hit

        return best_hit

    def _build_hit(self, score, parsed, motif_len, seq_type="dna", frame=None):
        core = parsed["core_len"]
        pct_id = (parsed["matches"] / core * 100) if core > 0 else 0.0
        coverage = (parsed["matches"] / motif_len * 100) if motif_len > 0 else 0.0
        norm_score = score / motif_len if motif_len > 0 else 0.0

        hit = {
            "score": score,
            "norm_score": round(norm_score, 2),
            "pct_id": round(pct_id, 2),
            "coverage": round(coverage, 2),
            "matches": parsed["matches"],
            "mismatches": parsed["mismatches"],
            "internal_gaps": parsed["ins"] + parsed["dels"],
            "alignment_len": core,
            "start_pos": parsed["start_query"],
            "end_pos": parsed["end_query"],
            "seq_type": seq_type,
        }
        if frame is not None:
            hit["protein_frame"] = frame
        return hit

    # ── Filtering & dedup ─────────────────────────────────────────────────

    @staticmethod
    def _min_pct_id_for_length(motif_len, base_min_pct_id=85.0):
        """Length-adaptive identity threshold."""
        penalty = 15.0 * math.exp(-motif_len / 100.0)
        return min(base_min_pct_id + penalty, 99.0)

    @staticmethod
    def _dedup_hits_per_token(hits, iou_threshold=0.5):
        """Per-token NMS: remove dominated hits overlapping the same region."""
        ranked = sorted(hits, key=lambda h: h.get("norm_score", 0), reverse=True)
        kept = []

        for hit in ranked:
            h_start = hit.get("dna_start", hit.get("start_pos", 0))
            h_end = hit.get("dna_end", hit.get("end_pos", 0))
            h_span = max(h_end - h_start + 1, 1)

            dominated = False
            for k in kept:
                k_start = k.get("dna_start", k.get("start_pos", 0))
                k_end = k.get("dna_end", k.get("end_pos", 0))
                k_span = max(k_end - k_start + 1, 1)

                overlap = max(0, min(h_end, k_end) - max(h_start, k_start) + 1)
                union = h_span + k_span - overlap
                iou = overlap / union if union > 0 else 0

                if iou >= iou_threshold:
                    dominated = True
                    break

            if not dominated:
                kept.append(hit)

        return kept

    @staticmethod
    def _dedup_hits_global(hits, iou_threshold=0.3):
        """Cross-token NMS: when different tokens claim the same region,
        keep the hit with higher norm_score."""
        ranked = sorted(hits, key=lambda h: h.get("norm_score", 0), reverse=True)
        kept = []

        for hit in ranked:
            h_start = hit.get("dna_start", 0)
            h_end = hit.get("dna_end", 0)
            h_span = max(h_end - h_start + 1, 1)

            dominated = False
            for k in kept:
                k_start = k.get("dna_start", 0)
                k_end = k.get("dna_end", 0)
                k_span = max(k_end - k_start + 1, 1)

                overlap = max(0, min(h_end, k_end) - max(h_start, k_start) + 1)
                union = h_span + k_span - overlap
                iou = overlap / union if union > 0 else 0

                if iou >= iou_threshold:
                    dominated = True
                    break

            if not dominated:
                kept.append(hit)

        return kept

    # ── Core scoring ──────────────────────────────────────────────────────

    def _run_tokens(
        self,
        target_dna,
        tokens_to_search,
        min_pct_id=85.0,
        min_norm_score=0.0,
        min_coverage=50.0,
        adaptive_id=True,
        kmer_prefilter=True,
        kmer_min_overlap=0.15,
        norm_score_prefilter=1.0,
    ):
        """Score motifs and return best hits per token, deduped by location."""
        target_len = len(target_dna)
        target_kmers = self._kmer_set(target_dna, k=15) if target_len >= 15 else set()
        protein_frames = self._get_six_frames(target_dna)

        token_set = set(tokens_to_search)
        sub_db = [m for m in self.motif_db if m["token"] in token_set]

        token_hits = defaultdict(list)

        for entry in sub_db:
            motif_seq = entry.get("sequence")
            if not motif_seq:
                continue

            is_protein = (entry.get("seq_type") or "dna").lower() == "protein"

            if not is_protein:
                motif_seq = self._clean_seq(motif_seq)
            else:
                motif_seq = re.sub(r"\s+", "", motif_seq).upper()

            if len(motif_seq) < 10:
                continue

            # Pass 1: K-mer pre-filter (DNA only)
            if kmer_prefilter and not is_protein and len(motif_seq) >= 15:
                motif_kmers = entry.get("_kmer_cache")
                if motif_kmers is None:
                    motif_kmers = self._kmer_set(motif_seq, k=15)
                    entry["_kmer_cache"] = motif_kmers
                if len(motif_kmers) > 0:
                    overlap = len(target_kmers & motif_kmers) / len(motif_kmers)
                    if overlap < kmer_min_overlap:
                        continue

            # Pass 2: Score-only fast check (DNA only)
            if not is_protein and norm_score_prefilter > 0:
                fast_norm = self._align_dna_score_only(target_dna, motif_seq)
                if fast_norm < norm_score_prefilter:
                    continue

            # Pass 3: Full trace alignment
            if is_protein:
                hit = self._align_protein(target_dna, protein_frames, motif_seq, target_len)
            else:
                hit = self._align_dna(target_dna, motif_seq)

            if hit is None:
                continue

            # Length-adaptive identity threshold
            if adaptive_id:
                effective_min_id = self._min_pct_id_for_length(
                    len(motif_seq), base_min_pct_id=min_pct_id
                )
            else:
                effective_min_id = min_pct_id

            if hit["pct_id"] < effective_min_id:
                continue
            if hit["norm_score"] < min_norm_score:
                continue
            if hit["coverage"] < min_coverage:
                continue

            # Add DNA coords for dedup
            if "dna_start" not in hit:
                hit["dna_start"] = hit["start_pos"]
                hit["dna_end"] = hit["end_pos"]

            hit["token"] = entry["token"]
            hit["db_entry"] = entry.get("sseqid", "")
            token_hits[entry["token"]].append(hit)

        # Dedup within each token
        all_hits = []
        for tok, hits in token_hits.items():
            deduped = self._dedup_hits_per_token(hits)
            all_hits.extend(deduped)

        # Global dedup across tokens
        all_hits = self._dedup_hits_global(all_hits)

        return all_hits

    # ── Public API (Scorer interface) ─────────────────────────────────────

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
        **run_kwargs,
    ) -> float:
        # Extract tokens from prompt
        expected_tokens = re.findall(r"<[^>]+>", prompt)
        expected_tokens = [
            t for t in expected_tokens
            if any(t.strip("<>").startswith(p) for p in
                   {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"})
        ]

        if not expected_tokens:
            return 0.0

        # Clean sequence
        raw = sequence.upper()
        target_dna = re.sub(r"<[^>]+>", "", raw)
        target_dna = re.sub(r"[^ATGCN]", "", target_dna)

        if len(target_dna) < 20:
            return 0.0

        result = self._score_composite(
            target_dna, expected_tokens,
            w_id=w_id, w_cov=w_cov, w_norm=w_norm,
            norm_score_cap=norm_score_cap, sharpness=sharpness,
            recall_floor=recall_floor, **run_kwargs,
        )
        return result["composite"]

    def score_sequence_detailed(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> dict[str, Any]:
        expected_tokens = re.findall(r"<[^>]+>", prompt)
        expected_tokens = [
            t for t in expected_tokens
            if any(t.strip("<>").startswith(p) for p in
                   {"AMR_", "ORI_", "PROM_", "REPORTER_", "REP_", "TAG_", "ELEM_"})
        ]

        raw = sequence.upper()
        target_dna = re.sub(r"<[^>]+>", "", raw)
        target_dna = re.sub(r"[^ATGCN]", "", target_dna)

        if not expected_tokens or len(target_dna) < 20:
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

        return self._score_composite(target_dna, expected_tokens, **kwargs)

    def _score_composite(
        self,
        target_dna,
        expected_tokens,
        w_id=0.4,
        w_cov=0.35,
        w_norm=0.25,
        norm_score_cap=5.0,
        sharpness=2.0,
        recall_floor=0.5,
        **run_kwargs,
    ):
        """Compute composite score for how well a plasmid matches expected annotations."""
        hits = self._run_tokens(target_dna, expected_tokens, **run_kwargs)

        best_per_token = {}
        for hit in hits:
            tok = hit["token"]
            if tok not in best_per_token or hit["norm_score"] > best_per_token[tok]["norm_score"]:
                best_per_token[tok] = hit

        token_scores = {}
        found_scores = []

        for tok in expected_tokens:
            if tok in best_per_token:
                h = best_per_token[tok]
                id_score = h["pct_id"] / 100.0
                cov_score = h["coverage"] / 100.0
                norm_scaled = min(h["norm_score"] / norm_score_cap, 1.0)

                quality = w_id * id_score + w_cov * cov_score + w_norm * norm_scaled

                token_scores[tok] = {
                    "score": round(quality, 4),
                    "found": True,
                    "pct_id": h["pct_id"],
                    "coverage": h["coverage"],
                    "norm_score": h["norm_score"],
                    "db_entry": h.get("db_entry", ""),
                    "dna_start": h.get("dna_start"),
                    "dna_end": h.get("dna_end"),
                }
                found_scores.append(quality)
            else:
                token_scores[tok] = {
                    "score": 0.0,
                    "found": False,
                    "pct_id": 0,
                    "coverage": 0,
                    "norm_score": 0,
                    "db_entry": None,
                    "dna_start": None,
                    "dna_end": None,
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
            "hits": hits,
        }
