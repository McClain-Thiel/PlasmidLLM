"""PyTorch Dataset and DataLoader for plasmid training pairs."""

from __future__ import annotations

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
        prompt_col = "token_prompt" if "token_prompt" in col_names else "prompt"
        self.prompts = table.column(prompt_col).to_pylist()
        completion_col = "token_completion" if "token_completion" in col_names else "sequence"
        self.completions = table.column(completion_col).to_pylist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt_str = self.prompts[idx]
        prompt_ids = self.tokenizer.encode(prompt_str)
        completion_ids = self.tokenizer.encode(self.completions[idx])

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


