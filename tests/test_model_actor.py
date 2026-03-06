"""Integration tests for ModelActor + Algorithm, using Ray + gpt2.

These tests require Ray and network access to download gpt2 (~500MB).
They run on CPU if no GPU is available. Mark with @pytest.mark.integration
so they can be skipped in CI with `pytest -m "not integration"`.
"""

import pytest
import torch

ray = pytest.importorskip("ray")
transformers = pytest.importorskip("transformers")

MODEL_ID = "gpt2"


@pytest.fixture(scope="module")
def ray_context():
    """Start Ray once for the whole module, shut down after."""
    ray.init(ignore_reinit_error=True, num_gpus=0)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def actor(ray_context):
    """Single ModelActor using gpt2 on CPU."""
    from post_training.common.model import ModelActor

    # Override num_gpus=0 so tests run on CPU
    ActorCls = ModelActor.options(num_gpus=0)
    a = ActorCls.remote(MODEL_ID, learning_rate=1e-4, bf16=False, max_steps=100)
    yield a


class TestSubstringScorer:
    """Verify our toy scorer satisfies the Scorer protocol."""

    def test_score_returns_tensor(self):
        from post_training.runners.run import SubstringScorer

        scorer = SubstringScorer(targets=["the", "and"])
        result = scorer.score(
            prompts=["a", "b"],
            completions=["the cat and dog", "nothing here"],
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)
        assert result[0].item() > result[1].item()


@pytest.mark.integration
class TestModelActorGetters:
    def test_get_step(self, actor):
        step = ray.get(actor.get_step.remote())
        assert step == 0

    def test_get_lr(self, actor):
        lr = ray.get(actor.get_lr.remote())
        assert lr > 0

    def test_get_vocab_size(self, actor):
        vs = ray.get(actor.get_vocab_size.remote())
        assert vs == 50257  # gpt2

    def test_get_pad_token_id(self, actor):
        pid = ray.get(actor.get_pad_token_id.remote())
        assert isinstance(pid, int)

    def test_get_weights_returns_dict(self, actor):
        w = ray.get(actor.get_weights.remote())
        assert isinstance(w, dict)
        assert len(w) > 0
        first_val = next(iter(w.values()))
        assert first_val.device == torch.device("cpu")


@pytest.mark.integration
class TestModelActorGeneration:
    def test_generate_returns_result(self, actor):
        from post_training.common.objects import GenerationResult

        gen = ray.get(actor.generate.remote(
            ["Hello world"], max_new_tokens=16,
        ))
        assert isinstance(gen, GenerationResult)
        assert len(gen.prompts) == 1
        assert len(gen.completion_texts) == 1
        assert gen.full_ids.shape[0] == 1
        assert gen.completion_ids.shape[1] <= 16

    def test_generate_batch(self, actor):
        gen = ray.get(actor.generate.remote(
            ["Hello", "World"], max_new_tokens=8,
        ))
        assert len(gen.prompts) == 2
        assert gen.full_ids.shape[0] == 2


@pytest.mark.integration
class TestModelActorLogProbs:
    def test_get_log_probs(self, actor):
        from post_training.common.objects import LogProbResult

        gen = ray.get(actor.generate.remote(["Test input"], max_new_tokens=8))
        lp = ray.get(actor.get_log_probs.remote(gen.full_ids, gen.prompt_len))

        assert isinstance(lp, LogProbResult)
        assert lp.per_token.shape[0] == 1
        assert lp.mask.shape == lp.per_token.shape
        assert lp.mean_per_seq.shape == (1,)
        assert (lp.per_token[lp.mask] <= 0).all()

    def test_ref_log_probs(self, actor):
        gen = ray.get(actor.generate.remote(["Test input"], max_new_tokens=8))
        policy_lp = ray.get(actor.get_log_probs.remote(
            gen.full_ids, gen.prompt_len, use_ref=False,
        ))
        ref_lp = ray.get(actor.get_log_probs.remote(
            gen.full_ids, gen.prompt_len, use_ref=True,
        ))

        # Before any training, policy == ref
        assert torch.allclose(policy_lp.per_token, ref_lp.per_token, atol=1e-4)


@pytest.mark.integration
class TestModelActorTraining:
    def test_forward_backward_and_step(self, actor):
        from post_training.common.objects import BackwardResult

        gen = ray.get(actor.generate.remote(["Train me"], max_new_tokens=8))
        lp = ray.get(actor.get_log_probs.remote(gen.full_ids, gen.prompt_len))

        B = gen.full_ids.shape[0]
        advantages = torch.ones(B)

        ray.get(actor.zero_grad.remote())
        result = ray.get(actor.forward_backward.remote(
            full_ids=gen.full_ids,
            prompt_len=gen.prompt_len,
            advantages=advantages,
            old_log_probs=lp.per_token,
            ref_log_probs=lp.per_token,
            mask=lp.mask,
            loss_fn_name="reinforce",
        ))
        assert "loss" in result
        assert isinstance(result["loss"], float)

        step_result = ray.get(actor.clip_and_step.remote())
        assert isinstance(step_result, BackwardResult)
        assert step_result.step == 1
        assert step_result.grad_norm >= 0

    def test_unknown_loss_raises(self, actor):
        gen = ray.get(actor.generate.remote(["Test"], max_new_tokens=4))
        lp = ray.get(actor.get_log_probs.remote(gen.full_ids, gen.prompt_len))

        ray.get(actor.zero_grad.remote())
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.forward_backward.remote(
                full_ids=gen.full_ids,
                prompt_len=gen.prompt_len,
                advantages=torch.ones(1),
                old_log_probs=lp.per_token,
                ref_log_probs=lp.per_token,
                mask=lp.mask,
                loss_fn_name="nonexistent_loss",
            ))


@pytest.mark.integration
class TestModelActorWeightSync:
    def test_load_weights(self, ray_context):
        from post_training.common.model import ModelActor

        ActorCls = ModelActor.options(num_gpus=0)
        a1 = ActorCls.remote(MODEL_ID, bf16=False)
        a2 = ActorCls.remote(MODEL_ID, bf16=False)

        w1 = ray.get(a1.get_weights.remote())
        ray.get(a2.load_weights.remote(w1))
        w2 = ray.get(a2.get_weights.remote())

        for key in w1:
            assert torch.equal(w1[key], w2[key]), f"Mismatch in {key}"

    def test_sync_ref_to_policy(self, actor):
        ray.get(actor.sync_ref_to_policy.remote())
        policy_w = ray.get(actor.get_weights.remote())
        ref_w = ray.get(actor.get_ref_weights.remote())

        for key in policy_w:
            assert torch.equal(policy_w[key], ref_w[key])


@pytest.mark.integration
class TestGRPOIntegration:
    """End-to-end: algorithm.step() with a real ModelActor."""

    def test_grpo_step(self, ray_context):
        from post_training.algorithms import GRPOAlgorithm
        from post_training.common.model import ModelActor
        from post_training.runners.run import SubstringScorer

        ActorCls = ModelActor.options(num_gpus=0)
        actor = ActorCls.remote(MODEL_ID, bf16=False, learning_rate=1e-4)

        algo = GRPOAlgorithm(
            kl_coef=0.1, cliprange=0.2,
            num_generations=2, micro_batch_size=4,
            gen_kwargs={"max_new_tokens": 16},
        )
        scorer = SubstringScorer(targets=["the"])

        metrics = algo.step([actor], ["Hello world", "Test prompt"], scorer)

        assert "loss" in metrics
        assert "mean_reward" in metrics
        assert "kl" in metrics
        assert "reward_best" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["mean_reward"], float)

        step = ray.get(actor.get_step.remote())
        assert step >= 1
