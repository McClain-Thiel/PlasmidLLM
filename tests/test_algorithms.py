"""Tests for Algorithm subclasses — advantage computation and helpers.

Requires ray (imported by algorithm modules), but no GPU or model loading.
"""

import pytest
import torch

ray = pytest.importorskip("ray")

from post_training.algorithms import GRPOAlgorithm, PPOAlgorithm, build_algorithm
from post_training.algorithms.base import Algorithm


class TestBuildAlgorithm:
    def test_build_grpo(self):
        algo = build_algorithm("grpo", kl_coef=0.3, cliprange=0.2, num_generations=4)
        assert isinstance(algo, GRPOAlgorithm)
        assert algo.kl_coef == 0.3

    def test_build_ppo(self):
        algo = build_algorithm("ppo", cliprange=0.1, ppo_epochs=2)
        assert isinstance(algo, PPOAlgorithm)
        assert algo.ppo_epochs == 2

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown algorithm"):
            build_algorithm("nonexistent")


class TestGRPOAdvantages:
    def test_group_normalization(self):
        algo = GRPOAlgorithm(num_generations=4)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        adv = algo.compute_advantages(rewards, group_size=4)

        assert adv.shape == (8,)
        # Each group of 4 should be z-scored independently
        group1 = adv[:4]
        group2 = adv[4:]
        assert abs(group1.mean().item()) < 1e-5
        assert abs(group2.mean().item()) < 1e-5

    def test_single_group(self):
        algo = GRPOAlgorithm(num_generations=3)
        rewards = torch.tensor([1.0, 2.0, 3.0])
        adv = algo.compute_advantages(rewards, group_size=3)
        assert adv.shape == (3,)
        assert abs(adv.mean().item()) < 1e-5

    def test_uniform_rewards_give_zero_advantage(self):
        algo = GRPOAlgorithm(num_generations=4)
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
        adv = algo.compute_advantages(rewards, group_size=4)
        assert torch.allclose(adv, torch.zeros(4), atol=1e-5)


class TestPPOAdvantages:
    def test_z_score_normalization(self):
        algo = PPOAlgorithm()
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        adv = algo.compute_advantages(rewards)

        assert adv.shape == (5,)
        assert abs(adv.mean().item()) < 1e-5
        assert abs(adv.std().item() - 1.0) < 0.2

    def test_single_reward_passthrough(self):
        algo = PPOAlgorithm()
        rewards = torch.tensor([3.14])
        adv = algo.compute_advantages(rewards)
        assert adv.item() == pytest.approx(3.14)


class TestSharding:
    def test_even_split(self):
        shards = Algorithm._shard(list(range(8)), 4)
        assert len(shards) == 4
        assert all(len(s) == 2 for s in shards)
        flat = [x for s in shards for x in s]
        assert flat == list(range(8))

    def test_uneven_split(self):
        shards = Algorithm._shard(list(range(7)), 3)
        assert len(shards) == 3
        lengths = [len(s) for s in shards]
        assert sum(lengths) == 7
        assert max(lengths) - min(lengths) <= 1

    def test_more_shards_than_items(self):
        shards = Algorithm._shard([1, 2], 5)
        assert len(shards) == 5
        non_empty = [s for s in shards if s]
        assert len(non_empty) == 2

    def test_single_shard(self):
        items = list(range(10))
        shards = Algorithm._shard(items, 1)
        assert shards == [items]
