"""PyTorch Dataset and DataLoader for plasmid training pairs."""

from __future__ import annotations

import re

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Subset


class PlasmidDataset(Dataset):
    """Dataset that reads prompt/completion pairs from parquet and tokenizes them.

    Each sample is: <prompt tokens> <SEP> <completion tokens> <EOS> padded/truncated to max_seq_len.
    Labels are the same as input_ids shifted right (for causal LM), with prompt tokens masked (-100).
    
    Args:
        parquet_path: Path to parquet file with prompt/completion columns
        tokenizer: Any tokenizer with encode() method and token ID properties
        max_seq_len: Maximum sequence length for padding/truncation
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_seq_len: int = 4096,
        circular_rotation: bool = False,
        soft_token_dropout: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if not 0 <= soft_token_dropout <= 1:
            raise ValueError("soft_token_dropout must be in [0, 1]")
        self.circular_rotation = circular_rotation
        self.soft_token_dropout = soft_token_dropout
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.sep_id = tokenizer.sep_token_id
        self.eos_id = tokenizer.eos_token_id

        # Read parquet — columns are lightweight tag strings + DNA, fits in RAM
        # Support both column naming conventions
        table = pq.read_table(parquet_path)
        col_names = table.column_names
        prompt_col = "token_prompt" if "token_prompt" in col_names else "prompt"
        self.prompts = table.column(prompt_col).to_pylist()
        completion_col = "token_completion" if "token_completion" in col_names else "sequence"
        self.completions = table.column(completion_col).to_pylist()
        self.lineage = {name: set(table.column(name).to_pylist())
                        for name in ("plasmid_id", "sequence_id", "cluster_id")
                        if name in col_names}

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt_str = self.prompts[idx]
        completion = self.completions[idx]
        if self.soft_token_dropout:
            # COPY is a requested biological attribute; keep it alongside hard tokens.
            prompt_str = re.sub(
                r"<(?:SP|VEC)_[^>]+>",
                lambda match: "" if torch.rand(()).item() < self.soft_token_dropout
                else match.group(), prompt_str,
            )
        if self.circular_rotation:
            if not completion or re.search("[^ACGTNacgtn]", completion):
                raise ValueError(f"Row {idx}: circular rotation requires nonempty DNA")
            offset = torch.randint(len(completion), ()).item()
            completion = completion[offset:] + completion[:offset]
        encode_fn = getattr(self.tokenizer, "encode_fast", self.tokenizer.encode)
        prompt_ids = encode_fn(prompt_str)
        completion_ids = encode_fn(completion)

        # Check if prompt already contains BOS and separator (<SEQ> or <SEP>)
        has_bos = prompt_ids and prompt_ids[0] == self.bos_id
        seq_token_id = self.tokenizer.convert_tokens_to_ids("<SEQ>") if "<SEQ>" in prompt_str else None
        has_sep = (seq_token_id is not None and seq_token_id in prompt_ids) or (
            prompt_ids and prompt_ids[-1] == self.sep_id
        )

        if has_bos and has_sep:
            # Prompt already formatted: <BOS>...<SEQ> — just append completion + EOS
            input_ids = prompt_ids + completion_ids + [self.eos_id]
            prompt_len = len(prompt_ids)
        else:
            # Legacy format: wrap with BOS + SEP
            input_ids = [self.bos_id] + prompt_ids + [self.sep_id] + completion_ids + [self.eos_id]
            prompt_len = 1 + len(prompt_ids) + 1  # BOS + prompt + SEP

        # Truncate to max_seq_len
        input_ids = input_ids[: self.max_seq_len]

        # Build labels: mask prompt with -100, predict completion tokens
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]

        # Pad
        seq_len = len(input_ids)
        pad_len = self.max_seq_len - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        input_ids = input_ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def validate_disjoint(train: PlasmidDataset, validation: PlasmidDataset) -> None:
    """Reject accidental source/sequence/cluster leakage in explicit partitions."""
    for key in ("plasmid_id", "sequence_id", "cluster_id"):
        if key not in train.lineage or key not in validation.lineage:
            raise ValueError(f"Explicit partitions require {key} provenance")
        overlap = train.lineage[key] & validation.lineage[key]
        if overlap:
            raise ValueError(f"Train/validation overlap on {key}: {len(overlap)} values")


def minority_sampling_weights(
    prompts: list[str], target_fraction: float = 0.1, max_weight: float = 4.0,
) -> torch.Tensor:
    """Capped square-root inverse frequency, fitted only on training prompts."""
    from pretraining.build_cluster_split import MINORITY_TOKENS

    if not prompts or not 0 < target_fraction < 1 or max_weight < 1:
        raise ValueError("Invalid minority sampler inputs")
    weights = torch.ones(len(prompts), dtype=torch.double)
    for token in MINORITY_TOKENS:
        mask = torch.tensor([token in prompt for prompt in prompts], dtype=torch.bool)
        count = int(mask.sum())
        if not count:
            raise ValueError(f"Training split contains no examples of {token}")
        weight = min(max_weight, max(1.0, (target_fraction * len(prompts) / count) ** 0.5))
        weights[mask] = torch.maximum(weights[mask], torch.tensor(weight, dtype=torch.double))
    return weights


def train_val_split(
    dataset: Dataset,
    val_split: float = 0.05,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Deterministic train/val split."""
    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_size = int(n * val_split)
    return Subset(dataset, indices[val_size:]), Subset(dataset, indices[:val_size])


