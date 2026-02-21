"""PyTorch Dataset and DataLoader for plasmid training pairs."""

from __future__ import annotations

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Subset


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
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.sep_id = tokenizer.sep_token_id
        self.eos_id = tokenizer.eos_token_id

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

        # Concatenate: BOS + prompt + SEP + completion + EOS
        input_ids = [self.bos_id] + prompt_ids + [self.sep_id] + completion_ids + [self.eos_id]

        # Truncate to max_seq_len
        input_ids = input_ids[: self.max_seq_len]

        # Build labels: mask BOS + prompt + SEP with -100, predict completion tokens
        prompt_len = 1 + len(prompt_ids) + 1  # BOS + prompt + SEP
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


def build_dataloaders(
    dataset: Dataset,
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


class PlasmidPromptsDataset(Dataset):
    """Dataset of prompts for RL training.
    
    Loads from training_pairs parquet and filters to has_hard_tokens=True.
    Returns prompts formatted for generation.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        filter_hard_tokens: bool = True,
    ):
        self.tokenizer = tokenizer

        # Load data
        table = pq.read_table(parquet_path)
        col_names = table.column_names

        # Check if has_hard_tokens column exists
        if "has_hard_tokens" in col_names and filter_hard_tokens:
            # Filter to only prompts with hard tokens
            has_hard = table.column("has_hard_tokens").to_pylist()
            indices = [i for i, h in enumerate(has_hard) if h]
            table = table.take(indices)

        # Extract prompt column (everything before <SEP>)
        if "token_prompt" in col_names:
            self.prompts = table.column("token_prompt").to_pylist()
        elif "full_text" in col_names:
            # Parse prompts from full_text: extract everything before <SEP>
            import re
            full_texts = table.column("full_text").to_pylist()
            self.prompts = []
            for text in full_texts:
                match = re.search(r"(.*?)<SEP>", text)
                if match:
                    self.prompts.append(match.group(1))
                else:
                    # Fallback: use everything before first DNA sequence
                    self.prompts.append(text.split("A")[0].split("T")[0].split("C")[0].split("G")[0])
        else:
            raise ValueError(f"No valid prompt column found. Available: {col_names}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        prompt = self.prompts[idx]
        # Add SEP token for generation
        prompt_with_sep = prompt + "<SEP>"
        
        return {
            "input_ids": torch.tensor(self.tokenizer.encode(prompt_with_sep), dtype=torch.long),
            "prompt": prompt,  # Keep original for reward calculation
        }
