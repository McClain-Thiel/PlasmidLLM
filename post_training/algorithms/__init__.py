"""RL algorithms that drive ModelActor through its getter/setter API."""

try:
    from post_training.algorithms.base import Algorithm, Scorer
    from post_training.algorithms.grpo import GRPOAlgorithm
    from post_training.algorithms.ppo import PPOAlgorithm

    ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {
        "ppo": PPOAlgorithm,
        "grpo": GRPOAlgorithm,
    }

    def build_algorithm(name: str, **kwargs) -> Algorithm:
        """Instantiate an algorithm by name."""
        if name not in ALGORITHM_REGISTRY:
            raise KeyError(
                f"Unknown algorithm '{name}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        return ALGORITHM_REGISTRY[name](**kwargs)

except ImportError:
    pass
