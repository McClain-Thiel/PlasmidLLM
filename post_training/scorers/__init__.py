"""Scorer implementations for post-training reward computation."""

from importlib import import_module

from post_training.scorers.base import Scorer

SCORER_REGISTRY = {
    "alignment": ("post_training.scorers.alignment", "AlignmentScorer"),
    "binary_motif": ("post_training.scorers.binary_motif", "BinaryMotifScorer"),
    "fractional_motif": (
        "post_training.scorers.fractional_motif",
        "FractionalMotifScorer",
    ),
    "motif": ("post_training.scorers.motif", "MotifScorer"),
    "plannotate": ("post_training.scorers.plannotate", "PlannotateScorer"),
    "valid": ("post_training.scorers.valid", "ValidPlasmidScorer"),
}

_EXPORTED_SCORERS = {class_name: module for module, class_name in SCORER_REGISTRY.values()}


def _resolve_scorer(target):
    if not isinstance(target, tuple):
        return target
    module_name, class_name = target
    return getattr(import_module(module_name), class_name)


def build_scorer(name: str, **kwargs) -> Scorer:
    """Instantiate a scorer by name."""
    if name not in SCORER_REGISTRY:
        raise KeyError(f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}")
    scorer_class = _resolve_scorer(SCORER_REGISTRY[name])
    return scorer_class(**kwargs)


def __getattr__(name: str):
    """Preserve class imports without loading every optional scorer dependency."""
    if name not in _EXPORTED_SCORERS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTED_SCORERS[name]), name)
