"""Binary motif scorer.

Uses the same plannotate BLAST infrastructure as ``PlannotateScorer``
but applies an all-or-nothing reward: the candidate gets full reward
(1.0) only when *every* expected annotation token from the prompt is
detected via BLAST, and zero reward otherwise.

This is useful for curriculum stages where partial credit is
undesirable and you want a hard gate on motif presence.
"""

from __future__ import annotations

from typing import Any

from post_training.scorers.plannotate import (
    PlannotateScorer,
    _clean_dna,
    _parse_hard_tokens,
)


class BinaryMotifScorer(PlannotateScorer):
    """All-or-nothing BLAST-based motif scorer.

    Inherits DB construction, token bridging, and BLAST execution from
    ``PlannotateScorer``.  Overrides scoring so the reward is binary:

    * **1.0** — every expected annotation token in the prompt has at
      least one BLAST hit above the identity / coverage thresholds.
    * **0.0** — any expected token is missing.

    Parameters ``min_pct_id`` and ``min_coverage`` act as quality gates
    on individual hits; a hit that falls below either threshold is
    treated as absent.
    """

    def __init__(
        self,
        *args,
        min_pct_id: float = 95.0,
        min_coverage: float = 80.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_pct_id = min_pct_id
        self.min_coverage = min_coverage

    # ------------------------------------------------ Scorer interface
    def score_sequence(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> float:
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
            return self._binary_result(expected_tokens, {})

        best_per_token = self._run_blast(seq, expected_tokens)
        return self._binary_result(expected_tokens, best_per_token)

    def score_batch(
        self,
        prompts: list[str],
        sequences: list[str],
        **kwargs,
    ) -> list[float]:
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

        rewards: list[float] = []
        for i, expected in enumerate(all_tokens):
            if not expected:
                rewards.append(0.0)
                continue
            per_token = batch_hits.get(f"q{i}", {})
            result = self._binary_result(expected, per_token)
            rewards.append(result["reward"])

        return rewards

    # ------------------------------------------------ binary logic
    def _binary_result(
        self,
        expected_tokens: list[str],
        best_per_token: dict[str, dict],
    ) -> dict[str, Any]:
        """Return 1.0 only if every expected token passes quality gates."""
        token_hits: dict[str, dict[str, Any]] = {}
        all_present = True

        for tok in expected_tokens:
            hit = best_per_token.get(tok)
            if hit is None:
                token_hits[tok] = {"found": False, "passed": False}
                all_present = False
                continue

            passes = (
                hit["pct_id"] >= self.min_pct_id
                and hit["coverage"] >= self.min_coverage
            )
            token_hits[tok] = {
                "found": True,
                "passed": passes,
                "pct_id": hit["pct_id"],
                "coverage": hit["coverage"],
                "bit_score": hit["bit_score"],
                "evalue": hit["evalue"],
                "sseqid": hit["sseqid"],
            }
            if not passes:
                all_present = False

        n_expected = len(expected_tokens)
        n_passed = sum(1 for v in token_hits.values() if v.get("passed"))
        reward = 1.0 if (all_present and n_expected > 0) else 0.0

        return {
            "reward": reward,
            "all_present": all_present,
            "expected": n_expected,
            "passed": n_passed,
            "token_hits": token_hits,
        }
