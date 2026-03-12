"""Scorer implementations for post-training reward computation."""

from post_training.scorers.base import Scorer
from post_training.scorers.alignment import AlignmentScorer
from post_training.scorers.motif import MotifScorer
from post_training.scorers.plannotate import PlannotateScorer
from post_training.scorers.valid import ValidPlasmidScorer

SCORER_REGISTRY = {
    "alignment": AlignmentScorer,
    "motif": MotifScorer,
    "plannotate": PlannotateScorer,
    "valid": ValidPlasmidScorer,
}


def build_scorer(name: str, **kwargs) -> Scorer:
    """Instantiate a scorer by name."""
    if name not in SCORER_REGISTRY:
        raise KeyError(
            f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}"
        )
    return SCORER_REGISTRY[name](**kwargs)
