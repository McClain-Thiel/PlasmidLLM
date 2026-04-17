"""Fractional motif scorer.

Like BinaryMotifScorer but gives partial credit: reward = n_passed / n_expected.
If the model gets 1/2 motifs correct, reward is 0.5. Gets 2/3, reward is 0.67.

This provides a denser learning signal for multi-token curriculum stages
where the all-or-nothing binary reward is too sparse.
"""

from __future__ import annotations

from typing import Any

from post_training.scorers.binary_motif import BinaryMotifScorer


class FractionalMotifScorer(BinaryMotifScorer):
    """Fractional BLAST-based motif scorer.

    Inherits everything from BinaryMotifScorer but overrides reward logic:
    reward = n_passed / n_expected (instead of 1.0 only if all passed).
    """

    def _binary_result(
        self,
        expected_tokens: list[str],
        best_per_token: dict[str, dict],
    ) -> dict[str, Any]:
        """Return fractional reward: n_passed / n_expected."""
        token_hits: dict[str, dict[str, Any]] = {}

        for tok in expected_tokens:
            hit = best_per_token.get(tok)
            if hit is None:
                token_hits[tok] = {"found": False, "passed": False}
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

        n_expected = len(expected_tokens)
        n_passed = sum(1 for v in token_hits.values() if v.get("passed"))
        reward = n_passed / max(n_expected, 1)

        return {
            "reward": reward,
            "all_present": n_passed == n_expected,
            "expected": n_expected,
            "passed": n_passed,
            "token_hits": token_hits,
        }
