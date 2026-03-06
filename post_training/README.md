# post_training

Distributed post-training for language models on Ray. Algorithm-agnostic by design.

The goal: test scorers independently to validate they reward real plasmids and penalize garbage, then plug-and-play any combination of **algorithm**, **model**, and **scorer** for RL fine-tuning. The infrastructure stays the same; only the pieces you swap change.

## Philosophy

The central idea is a clean separation between **infrastructure** and **algorithm**.

**The ModelActor is a dumb GPU worker.** It holds model weights, runs forward/backward passes, and exposes a flat getter/setter API. It has no idea what RL algorithm is being run. Think of it like a database — it stores state and executes operations, but has no opinion on what you do with the results.

**The Algorithm is the driver.** It composes the actor's primitives into a training step. It decides what to fetch (log-probs? entropy? raw logits?), how to compute advantages, how to structure micro-batches, and which loss function to apply. Different algorithms use different subsets of the actor's API, and the actor doesn't care.

This inverts the typical pattern where the actor calls the algorithm internally. Instead:

```
Algorithm.step()
  ├── actor.generate(prompts)          # get completions
  ├── scorer.score(completions)        # external reward
  ├── algo.compute_advantages(rewards) # algorithm-specific
  ├── actor.get_log_probs(...)         # fetch what we need
  ├── actor.get_log_probs(use_ref=True)
  ├── actor.zero_grad()
  ├── actor.forward_backward(...)      # loss by name from registry
  └── actor.clip_and_step()            # or average_and_apply across actors
```

The actor never imports an algorithm. The algorithm never subclasses the actor. They communicate through data.

## Project structure

```
post_training/
├── common/
│   ├── model.py        # ModelActor — Ray remote GPU worker
│   ├── objects.py       # GenerationResult, LogProbResult, etc.
│   ├── losses.py        # Loss registry + loss functions (reinforce, ppo, grpo)
│   └── utils.py         # average_gradients, wandb_log, timer
├── algorithms/
│   ├── base.py          # Algorithm ABC + Scorer protocol
│   ├── grpo.py          # GRPOAlgorithm
│   └── ppo.py           # PPOAlgorithm
├── scorers/
│   ├── base.py          # Scorer ABC (score_sequence / score_batch)
│   ├── alignment.py     # AlignmentScorer — Smith-Waterman score-ratio
│   └── motif.py         # MotifScorer — CIGAR-based parasail alignment
└── runners/
    └── run_grpo.py      # Example: GRPO training end-to-end
```

## Quick start

```bash
# Minimal GRPO run with a toy scorer (uses gpt2 by default)
python -m post_training.runners.run_grpo

# Real model, more steps, with wandb
python -m post_training.runners.run_grpo \
    --model McClain/PlasmidLM-kmer6-MoE \
    --steps 100 \
    --num-actors 2 \
    --num-generations 8 \
    --lr 1e-5 \
    --kl-coef 0.1 \
    --wandb --wandb-project plasmid-rl
```

## Actor API surface

The actor exposes three categories of methods:

**Getters** — return state without side effects:
`get_weights`, `get_ref_weights`, `get_gradients`, `get_optimizer_state`, `get_step`, `get_lr`, `get_vocab_size`, `get_pad_token_id`, `get_log_probs`, `get_entropy`, `get_logits`, `get_kl`

**Setters** — mutate state:
`load_weights`, `load_ref_weights`, `sync_ref_to_policy`, `set_gradients`, `load_optimizer_state`, `set_train`, `set_eval`

**Compute** — run operations:
`generate`, `tokenize`, `decode`, `zero_grad`, `forward_backward`, `clip_and_step`, `save_checkpoint`, `load_checkpoint`

An algorithm only calls the methods it needs. GRPO uses `get_log_probs` and `get_kl`. A future DPO implementation would use `get_logits` instead. The actor code doesn't change.

## Loss registry

Arbitrary callables can't be shipped through Ray serialization. Instead, loss functions are registered by name at import time:

```python
@register_loss("grpo")
def grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages, mask,
              *, cliprange=0.2, kl_coef=0.1, **kwargs):
    ...
```

The actor's `forward_backward` method takes a `loss_fn_name` string and `loss_fn_kwargs` dict. It looks up the function from the registry and calls it. Currently registered: `reinforce`, `ppo`, `grpo`.

## Algorithms

Built-in algorithms and a registry for lookup:

| Name   | Class            | Loss     | Description |
|--------|------------------|----------|-------------|
| `grpo` | `GRPOAlgorithm`  | `grpo`   | Group Relative Policy Optimization (DeepSeek-style). Generates N completions per prompt, z-normalizes rewards within each group, clipped surrogate + KL penalty. |
| `ppo`  | `PPOAlgorithm`   | `ppo`    | Proximal Policy Optimization. Single batch, multiple PPO epochs, clipped surrogate + entropy bonus. |

```python
from post_training.algorithms import build_algorithm

algo = build_algorithm("grpo", kl_coef=0.1, cliprange=0.2, num_generations=8)
```

## Scorers

Scorers are independent of the training infrastructure. They implement `score_sequence(prompt, sequence) -> float` and optionally `score_batch` for vectorized scoring. This means you can unit-test a scorer in isolation — feed it real plasmid sequences and confirm it returns high scores, feed it random DNA and confirm low scores — before ever connecting it to RL.

| Name        | Class             | Description |
|-------------|-------------------|-------------|
| `alignment` | `AlignmentScorer` | Smith-Waterman score-ratio using parasail. Scores DNA and protein (6-frame translation). Per-component scores capped at 1.0, optional EOS bonus. |
| `motif`     | `MotifScorer`     | CIGAR-based semi-global alignment via parasail. Three-pass: k-mer pre-filter, score-only alignment, full trace. Composite score = quality x recall penalty. |

```python
from post_training.scorers import build_scorer

scorer = build_scorer("motif", motif_registry=my_registry)
```

The `Scorer` protocol expected by algorithms is minimal — just `score(prompts, completions) -> torch.Tensor`. The scorer ABC in `scorers/base.py` uses `score_sequence` / `score_batch` instead. To bridge the two, wrap your scorer or add a `score` method that calls `score_batch` and converts to a tensor.

## Multi-actor training

Multiple actors train in parallel with gradient averaging:

```python
actors = [ModelActor.remote("gpt2") for _ in range(4)]
GRPOAlgorithm.sync_actors(actors)  # broadcast weights from actor[0]

# Inside algo.step():
# 1. Each actor generates from its shard of prompts
# 2. Each actor computes gradients locally
# 3. Driver averages gradients across actors
# 4. Each actor applies the same averaged gradient
# All actors stay in lockstep — no weight broadcast needed after init.
```

## Reference model management

The actor holds a frozen copy of the model for KL penalties. By default it stays at the initial checkpoint. The algorithm controls when to update it:

- `sync_ref_to_policy()` — copy current policy weights into the reference (useful at curriculum stage boundaries)
- `load_ref_weights(state_dict)` — set reference to arbitrary weights

## Adding a new algorithm

1. Write a loss function and register it:
   ```python
   @register_loss("my_algo")
   def my_loss(log_probs, old_log_probs, ref_log_probs, advantages, mask, **kwargs):
       ...
   ```

2. Subclass `Algorithm` and implement `step()` and `compute_advantages()`, composing whatever actor methods you need:
   ```python
   class MyAlgorithm(Algorithm):
       def compute_advantages(self, rewards, **kwargs):
           return rewards - rewards.mean()

       def step(self, actors, prompts, scorer):
           gen = ray.get(actors[0].generate.remote(prompts, max_new_tokens=64))
           rewards = scorer.score(prompts, gen.completion_texts)
           advantages = self.compute_advantages(rewards)
           # ... get log probs, forward_backward, step ...
           return {"loss": loss, "mean_reward": rewards.mean().item()}
   ```

3. Register it in `algorithms/__init__.py` and use it:
   ```python
   algo = build_algorithm("my_algo", hyperparam=0.1)
   metrics = algo.step(actors, prompts, scorer)
   ```

The actor is unchanged. The training loop is unchanged. Only the algorithm is new.

## Adding a new scorer

1. Subclass `Scorer` from `scorers/base.py`:
   ```python
   class MyScorer(Scorer):
       def score_sequence(self, prompt, sequence, **kwargs):
           return some_reward_float
   ```

2. Register it in `scorers/__init__.py`:
   ```python
   SCORER_REGISTRY["my_scorer"] = MyScorer
   ```

3. Test it standalone before plugging into RL:
   ```python
   scorer = build_scorer("my_scorer")
   assert scorer.score_sequence(real_prompt, real_plasmid) > 0.8
   assert scorer.score_sequence(real_prompt, random_dna) < 0.2
   ```

## Logging

### Console logging

All components use Python's `logging` module. Algorithms log at `INFO` level for step summaries (timing, rewards, loss) and `DEBUG` level for per-micro-batch details. The ModelActor logs at `DEBUG` for individual operations (generate, forward_backward, clip_and_step). To see debug output:

```python
logging.getLogger("post_training").setLevel(logging.DEBUG)
```

### wandb

wandb integration is opt-in. Pass `--wandb` to any runner to enable it:

```bash
python -m post_training.runners.run_grpo \
    --wandb \
    --wandb-project plasmid-rl \
    --wandb-run-name "grpo-motif-v2" \
    --wandb-entity my-team
```

All hyperparameters are logged to `wandb.config` at init. Each training step automatically logs metrics under a prefix (`grpo/`, `ppo/`):

| Metric | Description |
|--------|-------------|
| `loss` | Mean policy loss |
| `mean_reward` | Mean reward across all completions |
| `reward_std`, `reward_min`, `reward_max` | Reward distribution |
| `reward_best`, `reward_worst` | Mean of per-group best/worst (GRPO) |
| `kl` | Mean KL divergence from reference model |
| `grad_norm` | Gradient norm after clipping |
| `lr` | Current learning rate |
| `adv_mean`, `adv_std` | Advantage statistics (GRPO) |
| `mean_completion_len` | Mean completion length in characters |
| `time_generation`, `time_scoring`, `time_train`, `time_total` | Phase timings in seconds |

For custom scripts, wandb logging is automatic if a run is active — algorithms call `wandb_log()` internally:

```python
import wandb
wandb.init(project="my-project", config={...})

algo = build_algorithm("grpo", ...)
metrics = algo.step(actors, prompts, scorer)  # auto-logged to wandb
```

If wandb is not installed or no run is active, all logging calls are silent no-ops.

## Combinatorics

The point of this design is combinatorial flexibility. Once you trust your scorers, any run is just a choice of three things:

```
(algorithm, model, scorer)
```

Everything else is hyperparameters. The registries make this programmatic:

```python
for algo_name in ["grpo", "ppo"]:
    for scorer_name in ["alignment", "motif"]:
        algo = build_algorithm(algo_name, **algo_hparams)
        scorer = build_scorer(scorer_name, **scorer_kwargs)
        for step in range(n_steps):
            metrics = algo.step(actors, prompts, scorer)
```
