"""Tests for the PlasmidDataset."""

import json
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from plasmid_llm.data import PlasmidDataset, build_dataloaders, train_val_split
from plasmid_llm.tokenizer import PlasmidTokenizer


@pytest.fixture
def vocab_file(tmp_path):
    vocab = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<SEP>": 3,
        "<UNK>": 4,
        "<AMR_KANAMYCIN>": 5,
        "<ORI_F1>": 6,
    }
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab))
    return path


@pytest.fixture
def parquet_file(tmp_path):
    prompts = ["<BOS><AMR_KANAMYCIN>"] * 100
    completions = ["ATCGATCG"] * 100
    table = pa.table({"token_prompt": prompts, "token_completion": completions})
    path = tmp_path / "data.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def tokenizer(vocab_file):
    return PlasmidTokenizer(vocab_file)


@pytest.fixture
def dataset(parquet_file, tokenizer):
    return PlasmidDataset(str(parquet_file), tokenizer, max_seq_len=32)


class TestPlasmidDataset:
    def test_length(self, dataset):
        assert len(dataset) == 100

    def test_item_keys(self, dataset):
        item = dataset[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_shapes(self, dataset):
        item = dataset[0]
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"].shape == (32,)

    def test_padding(self, dataset):
        item = dataset[0]
        # Should have some padding at the end
        mask = item["attention_mask"]
        assert mask.sum() < 32  # not all positions should be real tokens

    def test_labels_mask_prompt(self, dataset):
        item = dataset[0]
        # First few positions (prompt + SEP) should be masked as -100
        labels = item["labels"]
        assert labels[0].item() == -100  # <BOS> is part of prompt
        assert labels[1].item() == -100  # <AMR_KANAMYCIN> is part of prompt
        assert labels[2].item() == -100  # <SEP> is part of prompt


class TestSplit:
    def test_train_val_split(self, dataset):
        train_ds, val_ds = train_val_split(dataset, val_split=0.1, seed=42)
        assert len(train_ds) + len(val_ds) == len(dataset)
        assert len(val_ds) == 10

    def test_deterministic(self, dataset):
        t1, v1 = train_val_split(dataset, val_split=0.1, seed=42)
        t2, v2 = train_val_split(dataset, val_split=0.1, seed=42)
        assert t1.indices == t2.indices

    def test_build_dataloaders(self, dataset):
        train_dl, val_dl = build_dataloaders(
            dataset, batch_size=8, val_split=0.1, seed=42, num_workers=0
        )
        batch = next(iter(train_dl))
        assert batch["input_ids"].shape[0] == 8
        assert batch["input_ids"].shape[1] == 32
