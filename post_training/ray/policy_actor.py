"""Ray GPU actor for policy model: generation, log-prob computation, and training.

Holds the policy model, frozen reference model, tokenizer, and optimizer.
All GPU work (generation, forward passes, gradient updates) happens here.
"""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import ray
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class PolicyActor:
    """GPU-resident actor managing the policy and reference models.

    Handles:
    - Generating rollouts (tokenize prompts → model.generate → decode)
    - Computing log probabilities under policy and reference models
    - Training steps (loss → backward → optimizer.step)
    - Checkpointing
    """

    def __init__(
        self,
        model_checkpoint: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 5000,
        max_grad_norm: float = 1.0,
        bf16: bool = True,
        max_completion_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.95,
        seed: int = 42,
    ):
        import sys
        # Ensure project root and src/ are on path for imports
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        src_dir = str(Path(project_root) / "src")
        for p in [project_root, src_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from plasmid_llm.models.hf_plasmid_lm import (
            PlasmidLMForCausalLM,
            PlasmidLMTokenizer,
        )

        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bf16 = bf16
        self.max_grad_norm = max_grad_norm
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Load policy model
        log.info(f"Loading policy model from {model_checkpoint}")
        self.model = PlasmidLMForCausalLM.from_pretrained(model_checkpoint)
        self.model.to(self.device)
        self.model.train()

        # Load tokenizer
        self.tokenizer = PlasmidLMTokenizer.from_pretrained(model_checkpoint)
        self.tokenizer.padding_side = "left"
        # Ensure pad_token string is set (HF base class checks this, not just pad_token_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<PAD>"

        # Frozen reference model (same GPU — 68MB is trivial)
        log.info("Creating frozen reference model")
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Linear warmup + cosine decay scheduler
        import math
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda,
        )

        self._step_count = 0
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"PolicyActor ready: {n_params:,} trainable params on {self.device}")

    def generate_rollouts(self, prompts: List[str]) -> Dict:
        """Generate completions and compute log probs for policy + ref model.

        Args:
            prompts: List of prompt strings (with <SEP> appended).

        Returns:
            Dict with keys:
                prompts: list[str] — input prompts
                completion_texts: list[str] — decoded completions
                completion_ids: tensor (batch, max_completion_len) — token ids
                log_probs: tensor (batch,) — sum of per-token log probs under policy
                ref_log_probs: tensor (batch,) — sum of per-token log probs under ref model
        """
        self.model.eval()

        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        prompt_ids = encoded["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Generate completions
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": self.top_p,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.top_k > 0:
                gen_kwargs["top_k"] = self.top_k

            output_ids = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=encoded["attention_mask"],
                **gen_kwargs,
            )

        # Split into completion tokens
        completion_ids = output_ids[:, prompt_len:]

        # Decode completions
        completion_texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=False
        )

        # Compute log probs under policy and reference
        full_ids = output_ids
        log_probs = self._compute_sequence_log_probs(self.model, full_ids, prompt_len)
        with torch.no_grad():
            ref_log_probs = self._compute_sequence_log_probs(
                self.ref_model, full_ids, prompt_len
            )

        self.model.train()

        return {
            "prompts": prompts,
            "completion_texts": completion_texts,
            "completion_ids": completion_ids.cpu(),
            "full_ids": full_ids.cpu(),
            "prompt_len": prompt_len,
            "log_probs": log_probs.cpu(),
            "ref_log_probs": ref_log_probs.cpu(),
        }

    def _compute_sequence_log_probs(
        self, model: torch.nn.Module, full_ids: torch.Tensor, prompt_len: int
    ) -> torch.Tensor:
        """Compute mean per-token log probs for completion tokens.

        Returns per-token average (not sum) so that loss/KL magnitudes are
        independent of sequence length, preventing gradient explosion on
        long completions.

        Args:
            model: PlasmidLMForCausalLM instance.
            full_ids: (batch, seq_len) full input+completion token ids.
            prompt_len: Number of prompt tokens (log probs start after this).

        Returns:
            (batch,) mean per-token log probs over completion tokens.
        """
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else torch.no_grad()
        with ctx:
            outputs = model(input_ids=full_ids)
            logits = outputs.logits  # (batch, seq_len, vocab)

        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, prompt_len - 1 : -1, :]  # (batch, completion_len, vocab)
        shift_labels = full_ids[:, prompt_len:]  # (batch, completion_len)

        log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)
        # Gather log probs for actual tokens
        token_log_probs = log_probs_all.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch, completion_len)

        # Mask padding tokens
        mask = shift_labels != self.tokenizer.pad_token_id
        token_log_probs = token_log_probs * mask.float()

        # Per-token mean (not sum) — keeps loss scale independent of seq length
        token_counts = mask.sum(dim=-1).clamp(min=1)
        return token_log_probs.sum(dim=-1) / token_counts  # (batch,)

    def train_step(self, rollout_batch: Dict) -> Dict:
        """Apply a single gradient update from pre-computed rollouts + advantages.

        Args:
            rollout_batch: Dict with keys:
                full_ids: (batch, seq_len) token ids
                prompt_len: int
                advantages: (batch,) advantage estimates
                ref_log_probs: (batch,) reference log probs
                algorithm_name: str — which algorithm to use
                algorithm_kwargs: dict — algorithm constructor kwargs

        Returns:
            Dict with training metrics.
        """
        from post_training.ray.algorithms import build_algorithm

        full_ids = rollout_batch["full_ids"].to(self.device)
        prompt_len = rollout_batch["prompt_len"]
        advantages = rollout_batch["advantages"].to(self.device)
        ref_log_probs = rollout_batch["ref_log_probs"].to(self.device)
        old_log_probs = rollout_batch["old_log_probs"].to(self.device)

        algorithm = build_algorithm(
            rollout_batch["algorithm_name"],
            **rollout_batch.get("algorithm_kwargs", {}),
        )

        self.model.train()

        # Recompute log probs under current policy (for fresh gradients)
        log_probs = self._compute_sequence_log_probs(self.model, full_ids, prompt_len)

        loss = algorithm.compute_loss(log_probs, old_log_probs, advantages, ref_log_probs)

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        self.optimizer.step()
        self.scheduler.step()
        self._step_count += 1

        # KL divergence estimate
        with torch.no_grad():
            kl = (log_probs - ref_log_probs).mean().item()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "kl": kl,
            "lr": self.scheduler.get_last_lr()[0],
            "step": self._step_count,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model, tokenizer, and training state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

        # Copy custom model/tokenizer code + special_tokens for self-containment
        project_root = Path(__file__).resolve().parent.parent.parent
        model_dir = project_root / "src" / "plasmid_llm" / "models" / "hf_plasmid_lm"
        for fname in ["special_tokens.txt"]:
            src = project_root / "data" / fname
            if src.exists():
                shutil.copy2(src, path / fname)
        for fname in ["configuration_plasmid_lm.py", "modeling_plasmid_lm.py", "tokenization_plasmid_lm.py"]:
            src = model_dir / fname
            if src.exists():
                shutil.copy2(src, path / fname)

        log.info(f"Checkpoint saved to {path}")

    def get_weights(self) -> Dict:
        """Return policy model state dict (for external checkpointing)."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
