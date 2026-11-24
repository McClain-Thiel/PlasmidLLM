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
├── config.py           # PostTrainingConfig dataclass
├── configs/
│   ├── grpo_reward_ablation.py               # v2.0 three-seed GRPO-v2 (F1 + R_ORF + R_tag)
│   ├── grpo_plannotate_anyscale.py           # legacy v1.0 recall-only reward config
│   ├── grpo_plannotate_v3_anyscale.py        # adds length penalty, longer schedule
│   └── grpo_plannotate_curriculum_anyscale.py  # motif→plannotate curriculum variant
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
│   ├── motif.py         # MotifScorer — CIGAR-based parasail alignment
│   ├── plannotate.py    # Legacy v1.0 batched BLAST scorer
│   └── plannotate_v2.py # v2.0 specificity- and ORF-aware scorer (F1 + R_ORF + R_tag)
└── runners/
    ├── run.py           # Config-driven runner (main entrypoint)
    └── run_grpo.py      # CLI wrapper for quick GRPO experiments
```

## Quick start

```bash
# Local smoke test (builds a temporary config from flags, toy scorer only)
python -m post_training.runners.run_grpo --model gpt2 --steps 10

# Canonical v2.0 three-seed GRPO (F1 + R_ORF + R_tag)
PLASMID_REWARD_VERSION=v2 PLASMID_SEED=44 python -m post_training.runners.run post_training/configs/grpo_reward_ablation.py
```

## Configuration

Runs are configured via Python files that export a `PostTrainingConfig` dataclass — the same pattern used by `pretraining/configs/`. Each config captures the full `(algorithm, model, scorer)` combination plus all hyperparameters.

```python
# post_training/configs/my_run.py
from post_training.config import PostTrainingConfig

config = PostTrainingConfig(
    # Model
    model="McClain/PlasmidLM-kmer6-MoE",
    bf16=True,
    num_actors=2,

    # Algorithm
    algorithm="grpo",
    kl_coef=0.1,
    cliprange=0.2,
    num_generations=8,

    # Scorer
    scorer="motif",
    scorer_kwargs={"motif_db_path": "data/motif_registry.parquet"},

    # Generation
    max_new_tokens=1024,

    # Prompts
    prompts_parquet="data/training_pairs_v4.parquet",
    prompt_batch_size=8,

    # Training
    steps=500,
    learning_rate=1e-5,
    checkpoint_every=100,
    checkpoint_dir="checkpoints/my_run",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_motif_v2",
)
```

Then run it:

```bash
python -m post_training.runners.run post_training/configs/my_run.py
```

### Config fields

| Group | Field | Default | Description |
|-------|-------|---------|-------------|
| **Model** | `model` | `"gpt2"` | HF model id or local path |
| | `bf16` | `True` | Use bfloat16 mixed precision |
| | `seed` | `42` | Random seed |
| **Actor** | `num_actors` | `1` | Number of Ray GPU workers |
| | `learning_rate` | `1e-5` | Optimizer learning rate |
| | `weight_decay` | `0.01` | AdamW weight decay |
| | `warmup_steps` | `100` | LR warmup steps |
| | `max_grad_norm` | `1.0` | Gradient clipping |
| **Algorithm** | `algorithm` | `"grpo"` | `"grpo"` or `"ppo"` |
| | `kl_coef` | `0.1` | KL penalty coefficient (GRPO) |
| | `cliprange` | `0.2` | PPO/GRPO clip range |
| | `num_generations` | `4` | Completions per prompt (GRPO) |
| | `entropy_coeff` | `0.01` | Entropy bonus (PPO) |
| | `ppo_epochs` | `4` | Epochs per batch (PPO) |
| | `micro_batch_size` | `64` | Micro-batch size for gradient accumulation |
| **Scorer** | `scorer` | `"substring"` | Registered scorer name; see `scorers/__init__.py` |
| | `scorer_kwargs` | `{}` | Kwargs passed to scorer constructor |
| **Generation** | `max_new_tokens` | `64` | Max tokens per completion |
| | `temperature` | `1.0` | Sampling temperature |
| | `top_p` | `0.95` | Nucleus sampling threshold |
| **Prompts** | `prompts_parquet` | `None` | Path to parquet with prompt column |
| | `prompts_list` | `None` | Inline list of prompts |
| | `prompt_batch_size` | `4` | Prompts per training step |
| | `filter_hard_tokens` | `True` | Filter to prompts with hard tokens |
| **Training** | `steps` | `100` | Total training steps |
| | `checkpoint_every` | `50` | Save checkpoint every N steps |
| | `checkpoint_dir` | `"checkpoints/post_training"` | Checkpoint output directory |
| **Logging** | `wandb_project` | `None` | wandb project (set to enable wandb) |
| | `wandb_run_name` | `None` | wandb run name (auto-generated if omitted) |
| | `wandb_entity` | `None` | wandb team/entity |

### Sample configs

| Config | Algorithm | Scorer | Model | Purpose |
|--------|-----------|--------|-------|---------|
| `grpo_reward_ablation.py` | GRPO | `plannotate_v2` | `cluster_base_s42` (`PlasmidLM-kmer6` v2.0) | Released `v2.0` 3-seed GRPO-v2 (`McClain/PlasmidLM-kmer6-GRPO-plannotate`) |
| `grpo_plannotate_anyscale.py` | GRPO | `plannotate` | `hf_base` (`v1.0`) | Legacy `v1.0` recall-only GRPO checkpoint (`v1` branch) |
| `grpo_plannotate_v3_anyscale.py` | GRPO | `plannotate` | `PlasmidLM-kmer6` | Adds length penalty to discourage sequence drift |
| `grpo_plannotate_curriculum_anyscale.py` | GRPO | `plannotate` | binary_motif checkpoint | Motif→plannotate curriculum variant (ablation) |

The released step-800 checkpoint used `norm_score_cap=5.0` and no length penalty.
`grpo_plannotate_anyscale.py` pins that historical value explicitly; later exploratory
configs use a `1.5` cap and may add a length penalty.

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
| `plannotate` | `PlannotateScorer` | Legacy `v1.0` batched nucleotide/protein BLAST against `motif_registry_combined.parquet`. |
| `plannotate_v2` | `PlannotateV2Scorer` | **Primary `v2.0` reward** (`0.6 F1 + 0.3 R_ORF + 0.1 R_tag`) using parallel `blastn`/`blastx`/`blastp` (`90%` id, `80%` ref cov) + complete Table-11 circular ORFs & exact peptide tags. |
| `binary_motif` / `fractional_motif` | Curriculum scorers | Exact or partial motif placement rewards used in ablations. |
| `valid` | `ValidPlasmidScorer` | Sequence-level validity checks. |

```python
from post_training.scorers import build_scorer

scorer = build_scorer("motif", motif_db_path="data/motif_registry.parquet")
```

The `Scorer` base class exposes `score_sequence`, `score_batch`, and the
tensor-returning `score` method expected by the training algorithms.

## Multi-actor training

Multiple actors train in parallel with gradient averaging:

```python
config = PostTrainingConfig(
    model="McClain/PlasmidLM-kmer6-MoE",
    num_actors=4,
    ...
)
```

Under the hood:
1. Each actor generates from its shard of prompts
2. Each actor computes gradients locally
3. Driver averages gradients across actors
4. Each actor applies the same averaged gradient

All actors stay in lockstep — no weight broadcast needed after init.

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

3. Register it in `algorithms/__init__.py` and add an `algorithm_kwargs()` case to `PostTrainingConfig`.

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

4. Use it in a config:
   ```python
   config = PostTrainingConfig(
       scorer="my_scorer",
       scorer_kwargs={"threshold": 0.5},
       ...
   )
   ```

## Logging

### Console logging

All components use Python's `logging` module. Algorithms log at `INFO` level for step summaries (timing, rewards, loss) and `DEBUG` level for per-micro-batch details. The ModelActor logs at `DEBUG` for individual operations (generate, forward_backward, clip_and_step). To see debug output:

```python
logging.getLogger("post_training").setLevel(logging.DEBUG)
```

### wandb

wandb integration is opt-in. Set `wandb_project` in your config to enable it:

```python
config = PostTrainingConfig(
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_motif_v2",
    wandb_entity="my-team",
    ...
)
```

Or with the CLI wrapper:

```bash
python -m post_training.runners.run_grpo --wandb --wandb-project plasmid-rl
```

All config fields are logged to `wandb.config` at init. Each training step automatically logs metrics under a prefix (`grpo/`, `ppo/`):

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

If no W&B project is configured, W&B logging is disabled. If a project is
configured, the runner requires the W&B dependency and initializes the run
before allocating Ray actors.

## Combinatorics

The point of this design is combinatorial flexibility. Once you trust your scorers, any run is just a config file picking three things:

```python
config = PostTrainingConfig(
    algorithm="grpo",                    # ← swap algorithm
    model="McClain/PlasmidLM-kmer6-MoE", # ← swap model
    scorer="motif",                       # ← swap scorer
    ...
)
```

Everything else is hyperparameters. To sweep, just write multiple configs or generate them programmatically:

```python
from post_training.config import PostTrainingConfig
from post_training.runners.run import run

for algo in ["grpo", "ppo"]:
    for scorer in ["alignment", "motif"]:
        cfg = PostTrainingConfig(
            algorithm=algo,
            scorer=scorer,
            model="McClain/PlasmidLM-kmer6-MoE",
            wandb_project="PlasmidLLM",
            wandb_run_name=f"{algo}_{scorer}",
            ...
        )
        run(cfg)
```
