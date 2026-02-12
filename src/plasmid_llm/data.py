"""PyTorch Dataset and DataLoader for plasmid training pairs."""

from __future__ import annotations

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from plasmid_llm.tokenizer import PlasmidTokenizer


class PlasmidDataset(Dataset):
    """Dataset that reads prompt/completion pairs from parquet and tokenizes them.

    Each sample is: <prompt tokens> <SEP> <completion tokens> padded/truncated to max_seq_len.
    Labels are the same as input_ids shifted right (for causal LM), with prompt tokens masked (-100).
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: PlasmidTokenizer,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id

        # Read parquet — columns are lightweight tag strings + DNA, fits in RAM
        # Support both column naming conventions
        table = pq.read_table(parquet_path)
        col_names = table.column_names
        self.prompts = table.column("token_prompt").to_pylist()
        completion_col = "token_completion" if "token_completion" in col_names else "sequence"
        self.completions = table.column(completion_col).to_pylist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(self.prompts[idx])
        completion_ids = self.tokenizer.encode(self.completions[idx])

        # Concatenate: prompt + SEP + completion
        input_ids = prompt_ids + [self.sep_id] + completion_ids

        # Truncate to max_seq_len
        input_ids = input_ids[: self.max_seq_len]

        # Build labels: mask prompt portion with -100, predict completion tokens
        prompt_len = len(prompt_ids) + 1  # +1 for SEP
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


def train_val_split(
    dataset: PlasmidDataset,
    val_split: float = 0.05,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Deterministic train/val split."""
    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_size = int(n * val_split)
    return Subset(dataset, indices[val_size:]), Subset(dataset, indices[:val_size])


def build_dataloaders(
    dataset: PlasmidDataset,
    batch_size: int = 32,
    val_split: float = 0.05,
    seed: int = 42,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    train_ds, val_ds = train_val_split(dataset, val_split, seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
