"""ModelActor — a pure GPU worker with a getter/setter API.

The actor knows nothing about RL algorithms.  It exposes primitives:
  - generate text
  - compute log-probs (policy or ref)
  - compute entropy
  - get/set weights, optimizer state, ref model
  - forward pass → logits
  - backward from an externally-computed scalar loss
  - step optimizer, clip grads

An Algorithm class composes these primitives however it wants.
The actor is to the algorithm what a database is to an application —
it holds state and executes operations, but has no opinion on workflow.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from post_training.common.losses import LOSS_REGISTRY

# Register local model classes so from_pretrained uses our fixed code
# instead of downloading (potentially stale) remote code from HuggingFace.
from plasmid_llm.models.hf_plasmid_lm.configuration_plasmid_lm import PlasmidLMConfig
from plasmid_llm.models.hf_plasmid_lm.modeling_plasmid_lm import PlasmidLMForCausalLM

AutoConfig.register("plasmid_lm", PlasmidLMConfig)
AutoModelForCausalLM.register(PlasmidLMConfig, PlasmidLMForCausalLM)
from post_training.common.objects import (
    BackwardResult,
    EntropyResult,
    GenerationResult,
    LogitsResult,
    LogProbResult,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# ModelActor — pure GPU worker
# ══════════════════════════════════════════════════════════════════════════


@ray.remote(num_gpus=1)
class ModelActor:
    """Stateful GPU worker.  Every method is a getter, setter, or a thin
    compute primitive.  Zero RL logic lives here.

    Philosophy: if an algorithm needs some quantity from the model
    (log-probs, entropy, logits, KL, gradients), there should be a
    method that returns exactly that.  The algorithm composes them.
    """

    def __init__(
        self,
        model_id: str,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 5000,
        max_grad_norm: float = 1.0,
        bf16: bool = True,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bf16 = bf16
        self.max_grad_norm = max_grad_norm

        # Check bf16 hardware support; fall back to fp16 if unavailable
        if bf16 and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                log.warning("bf16 not supported on %s, falling back to fp16",
                            torch.cuda.get_device_name(self.device))
                self.bf16 = False
                self._use_fp16 = True
            else:
                self._use_fp16 = False
        else:
            self._use_fp16 = False

        # If model_id is an S3 path, download it locally first
        if model_id.startswith("s3://"):
            import subprocess
            local_path = Path("/tmp/model_checkpoint")
            if not local_path.exists():
                log.info("Downloading model from %s", model_id)
                subprocess.check_call(["aws", "s3", "sync", model_id, str(local_path)])
            model_id = str(local_path)

        log.info("Loading %s on %s (bf16=%s, fp16=%s)", model_id, self.device,
                 self.bf16, self._use_fp16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map={"": self.device},
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,  # tokenizer still needs remote code
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<PAD>"
        self._pad_id = self.tokenizer.pad_token_id

        self.ref_model = copy.deepcopy(self.model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # torch.compile with reduce-overhead mode uses CUDA graphs to
        # eliminate Python/kernel-launch overhead — major win for small models.
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.ref_model = torch.compile(self.ref_model, mode="reduce-overhead")
            log.info("torch.compile(mode='reduce-overhead') applied to policy and ref models")
        except Exception as e:
            log.warning("torch.compile failed, falling back to eager mode: %s", e)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self._step = 0

        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("Ready: %s trainable params on %s", f"{n:,}", self.device)

    # ══════════════════════════════════════════════════════════════════════
    # GETTERS — read state, no side effects
    # ══════════════════════════════════════════════════════════════════════

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_ref_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.ref_model.state_dict().items()}

    def get_optimizer_state(self) -> dict:
        state = self.optimizer.state_dict()
        cpu_state = copy.deepcopy(state)
        for group in cpu_state.get("state", {}).values():
            for k, v in group.items():
                if isinstance(v, torch.Tensor):
                    group[k] = v.cpu()
        return cpu_state

    def get_step(self) -> int:
        return self._step

    def get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size

    def get_pad_token_id(self) -> int:
        return self._pad_id

    def get_gradients(self) -> dict[str, torch.Tensor]:
        """Current .grad tensors on CPU (for multi-actor gradient averaging)."""
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu()
        return grads

    def get_gpu_stats(self) -> dict[str, float]:
        """Snapshot of GPU memory, utilization, power, and temperature."""
        if not torch.cuda.is_available():
            return {}

        dev = self.device
        stats: dict[str, float] = {
            "memory_allocated_gb": torch.cuda.memory_allocated(dev) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(dev) / 1e9,
            "memory_peak_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
        }

        props = torch.cuda.get_device_properties(dev)
        total = props.total_memory
        stats["memory_total_gb"] = total / 1e9
        stats["memory_utilization"] = torch.cuda.memory_allocated(dev) / total

        try:
            import pynvml
            pynvml.nvmlInit()
            idx = dev.index if dev.index is not None else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["compute_utilization"] = util.gpu / 100.0
            stats["mem_controller_utilization"] = util.memory / 100.0

            try:
                stats["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                pass
            try:
                stats["temp_c"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except pynvml.NVMLError:
                pass
        except (ImportError, Exception):
            pass

        return stats

    def reset_peak_memory(self) -> None:
        """Reset CUDA peak-memory tracker so next phase starts fresh."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    # ══════════════════════════════════════════════════════════════════════
    # SETTERS — mutate state
    # ══════════════════════════════════════════════════════════════════════

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )

    def load_ref_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.ref_model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )

    def sync_ref_to_policy(self) -> None:
        """Copy current policy → reference model."""
        self.ref_model.load_state_dict(self.model.state_dict())

    def load_optimizer_state(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def set_gradients(self, grads: dict[str, torch.Tensor]) -> None:
        """Overwrite .grad tensors (e.g. averaged grads from driver)."""
        for name, param in self.model.named_parameters():
            if name in grads:
                param.grad = grads[name].to(self.device)

    def set_train(self) -> None:
        self.model.train()

    def set_eval(self) -> None:
        self.model.eval()

    # ══════════════════════════════════════════════════════════════════════
    # COMPUTE — read-only operations on the model
    # ══════════════════════════════════════════════════════════════════════

    def generate(self, prompts: list[str], **gen_kwargs) -> GenerationResult:
        """Sample completions using fast generate_simple().  No log-probs, no gradients."""
        t0 = time.monotonic()
        log.debug("generate: %d prompt(s), gen_kwargs=%s", len(prompts), gen_kwargs)
        self.model.eval()
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        prompt_len = encoded["input_ids"].shape[1]

        # Map gen_kwargs to generate_simple() parameters
        max_new_tokens = gen_kwargs.pop("max_new_tokens", 2500)
        temperature = gen_kwargs.pop("temperature", 0.3)
        top_p = gen_kwargs.pop("top_p", 0.95)
        top_k = gen_kwargs.pop("top_k", 0)
        eos_token_id = gen_kwargs.pop("eos_token_id", self.tokenizer.eos_token_id)
        # Pop HF-specific kwargs that don't apply to generate_simple
        gen_kwargs.pop("do_sample", None)
        gen_kwargs.pop("pad_token_id", None)

        # Access the underlying model (unwrap torch.compile if needed)
        model = self.model
        underlying = getattr(model, "_orig_mod", model)

        with torch.no_grad(), self._autocast():
            output_ids = underlying.generate_simple(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
                pad_token_id=self._pad_id,
            )

        self.model.train()
        comp_ids = output_ids[:, prompt_len:]
        comp_lens = (comp_ids != self._pad_id).sum(-1).float()
        elapsed = time.monotonic() - t0
        total_tokens = comp_lens.sum().item()
        tok_per_sec = total_tokens / max(elapsed, 1e-6)
        log.info(
            "generate: %.2fs — %d seqs, prompt_len=%d, mean_comp=%.1f, %.0f tok/s",
            elapsed, comp_ids.shape[0], prompt_len, comp_lens.mean().item(), tok_per_sec,
        )
        return GenerationResult(
            prompts=prompts,
            completion_texts=self.tokenizer.batch_decode(
                comp_ids, skip_special_tokens=False,
            ),
            completion_ids=comp_ids.cpu(),
            full_ids=output_ids.cpu(),
            prompt_len=prompt_len,
            elapsed_s=elapsed,
        )

    def tokenize(self, texts: list[str], **kwargs) -> dict:
        """Tokenize on the actor (useful when tokenizer isn't on driver)."""
        defaults = dict(return_tensors="pt", padding=True, truncation=True)
        defaults.update(kwargs)
        encoded = self.tokenizer(texts, **defaults)
        return {k: v.cpu() for k, v in encoded.items()}

    def decode(self, ids: torch.Tensor, **kwargs) -> list[str]:
        defaults = dict(skip_special_tokens=False)
        defaults.update(kwargs)
        return self.tokenizer.batch_decode(ids, **defaults)

    def get_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        *,
        use_ref: bool = False,
    ) -> LogProbResult:
        """Per-token log-probs for completion tokens.  No gradient."""
        model = self.ref_model if use_ref else self.model
        full_ids = full_ids.to(self.device)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            lp, mask = self._completion_log_probs(model, full_ids, prompt_len)
        if was_training:
            model.train()

        counts = mask.sum(-1).clamp(min=1)
        return LogProbResult(
            per_token=lp.cpu(),
            mask=mask.cpu(),
            mean_per_seq=(lp.sum(-1) / counts).cpu(),
        )

    def get_entropy(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        *,
        use_ref: bool = False,
    ) -> EntropyResult:
        """Per-token entropy for completion tokens.  No gradient."""
        model = self.ref_model if use_ref else self.model
        full_ids = full_ids.to(self.device)
        attention_mask = (full_ids != self._pad_id).long()

        was_training = model.training
        model.eval()
        with torch.no_grad(), self._autocast():
            logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
        if was_training:
            model.train()

        shift_logits = logits[:, prompt_len - 1 : -1, :].float()
        shift_labels = full_ids[:, prompt_len:]
        mask = shift_labels != self._pad_id

        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1) * mask.float()

        counts = mask.sum(-1).clamp(min=1)
        return EntropyResult(
            per_token=entropy.cpu(),
            mask=mask.cpu(),
            mean_per_seq=(entropy.sum(-1) / counts).cpu(),
        )

    def get_logits(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        *,
        use_ref: bool = False,
    ) -> LogitsResult:
        """Raw logits for completion tokens (for DPO, KTO, custom losses)."""
        model = self.ref_model if use_ref else self.model
        full_ids = full_ids.to(self.device)
        attention_mask = (full_ids != self._pad_id).long()

        was_training = model.training
        model.eval()
        with torch.no_grad(), self._autocast():
            logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
        if was_training:
            model.train()

        shift_logits = logits[:, prompt_len - 1 : -1, :]
        shift_labels = full_ids[:, prompt_len:]
        mask = shift_labels != self._pad_id

        return LogitsResult(logits=shift_logits.cpu().float(), mask=mask.cpu())

    def get_kl(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> dict[str, torch.Tensor]:
        """KL(π ‖ π_ref) per sequence — Schulman unbiased estimator."""
        full_ids = full_ids.to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            policy_lp, mask = self._completion_log_probs(
                self.model, full_ids, prompt_len,
            )
            ref_lp, _ = self._completion_log_probs(
                self.ref_model, full_ids, prompt_len,
            )
        if was_training:
            self.model.train()

        log_ratio = ref_lp - policy_lp
        pt_kl = (torch.exp(log_ratio) - log_ratio - 1.0) * mask.float()
        counts = mask.sum(-1).clamp(min=1)

        return {
            "per_token_kl": pt_kl.cpu(),
            "kl_per_seq": (pt_kl.sum(-1) / counts).cpu(),
            "mask": mask.cpu(),
        }

    # ══════════════════════════════════════════════════════════════════════
    # GRADIENT OPERATIONS — the only methods that mutate model params
    # ══════════════════════════════════════════════════════════════════════

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def forward_backward(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: torch.Tensor,
        loss_fn_name: str,
        loss_fn_kwargs: dict[str, Any] | None = None,
        *,
        accumulation_scale: float = 1.0,
    ) -> dict[str, float]:
        """Forward → loss → backward.  Does NOT step the optimizer.

        The loss function is looked up by name in LOSS_REGISTRY so the
        actor stays algorithm-agnostic.  Algorithms register their loss
        functions at import time via @register_loss.

        Args:
            accumulation_scale: Divide loss before .backward()
                (for gradient accumulation across micro-batches).
        """
        if loss_fn_name not in LOSS_REGISTRY:
            raise KeyError(
                f"Unknown loss '{loss_fn_name}'. "
                f"Available: {list(LOSS_REGISTRY.keys())}"
            )

        log.debug(
            "forward_backward: B=%d, prompt_len=%d, loss_fn=%s, accum_scale=%.1f",
            full_ids.shape[0], prompt_len, loss_fn_name, accumulation_scale,
        )

        self.model.train()
        full_ids = full_ids.to(self.device)
        advantages = advantages.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        ref_log_probs = ref_log_probs.to(self.device)
        mask = mask.to(self.device)

        lp, _ = self._completion_log_probs(self.model, full_ids, prompt_len)

        loss = LOSS_REGISTRY[loss_fn_name](
            log_probs=lp,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            mask=mask,
            **(loss_fn_kwargs or {}),
        )

        (loss / accumulation_scale).backward()
        log.debug("forward_backward: loss=%.6f", loss.item())
        return {"loss": loss.item()}

    def clip_and_step(self) -> BackwardResult:
        """Clip gradients → optimizer.step() → scheduler.step()."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self._step += 1

        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        lr = self.scheduler.get_last_lr()[0]
        log.debug("clip_and_step: step=%d grad_norm=%.4f lr=%.2e", self._step, gn, lr)

        return BackwardResult(grad_norm=gn, lr=lr, step=self._step)

    # ══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: str, s3_dest: str | None = None) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        torch.save(self.optimizer.state_dict(), out / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), out / "scheduler.pt")
        torch.save({"step": self._step}, out / "training_state.pt")
        log.info("Checkpoint → %s (step %d)", out, self._step)
        if s3_dest:
            self._sync_to_s3(str(out), s3_dest)

    def _sync_to_s3(self, local_dir: str, s3_dest: str) -> None:
        """Sync a local directory to S3 (runs on worker node)."""
        import subprocess
        log.info("Syncing %s → %s", local_dir, s3_dest)
        subprocess.call(["aws", "s3", "sync", local_dir, s3_dest, "--quiet"])

    def load_checkpoint(self, path: str) -> None:
        """Resume from a checkpoint saved by save_checkpoint()."""
        p = Path(path)
        loaded = AutoModelForCausalLM.from_pretrained(
            p, torch_dtype=torch.float32, device_map={"": self.device},
        )
        self.model.load_state_dict(loaded.state_dict())
        del loaded

        opt_path = p / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=True),
            )
        sched_path = p / "scheduler.pt"
        if sched_path.exists():
            self.scheduler.load_state_dict(
                torch.load(sched_path, map_location=self.device, weights_only=True),
            )
        state_path = p / "training_state.pt"
        if state_path.exists():
            self._step = torch.load(
                state_path, map_location="cpu", weights_only=True,
            )["step"]
        log.info("Resumed from %s at step %d", p, self._step)

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ══════════════════════════════════════════════════════════════════════

    def _completion_log_probs(
        self,
        model: torch.nn.Module,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probs for the completion portion of sequences.

        Gradient-transparent: callers control torch.no_grad() context.
        forward_backward() calls this WITH gradients; getters wrap in no_grad.
        """
        attention_mask = (full_ids != self._pad_id).long()
        with self._autocast():
            logits = model(input_ids=full_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, prompt_len - 1 : -1, :]
        shift_labels = full_ids[:, prompt_len:]

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        mask = shift_labels != self._pad_id
        return token_lp * mask.float(), mask

    def _autocast(self):
        if self.bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if getattr(self, "_use_fp16", False):
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()
