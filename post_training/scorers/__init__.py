"""Scorer implementations for post-training reward computation."""

from post_training.scorers.base import Scorer
from post_training.scorers.alignment import AlignmentScorer
from post_training.scorers.motif import MotifScorer

SCORER_REGISTRY = {
    "alignment": AlignmentScorer,
    "motif": MotifScorer,
}


def build_scorer(name: str, **kwargs) -> Scorer:
    """Instantiate a scorer by name."""
    if name not in SCORER_REGISTRY:
        raise KeyError(
            f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}"
        )
    return SCORER_REGISTRY[name](**kwargs)
