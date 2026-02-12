"""Tests for the PlasmidTokenizer."""

import json
import tempfile
from pathlib import Path

import pytest

from plasmid_llm.tokenizer import PlasmidTokenizer, _split_tokens


@pytest.fixture
def vocab_file(tmp_path):
    """Create a minimal vocabulary file."""
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
def tokenizer(vocab_file):
    return PlasmidTokenizer(vocab_file)


class TestSplitTokens:
    def test_tags_only(self):
        assert _split_tokens("<BOS><SEP>") == ["<BOS>", "<SEP>"]

    def test_dna_only(self):
        assert _split_tokens("ATCG") == ["A", "T", "C", "G"]

    def test_mixed(self):
        result = _split_tokens("<BOS><AMR_KANAMYCIN>ATCG")
        assert result == ["<BOS>", "<AMR_KANAMYCIN>", "A", "T", "C", "G"]

    def test_empty(self):
        assert _split_tokens("") == []


class TestTokenizer:
    def test_vocab_size(self, tokenizer):
        # 7 special tokens + 10 DNA bases (ATCGNatcgn)
        assert tokenizer.vocab_size == 17

    def test_special_token_ids(self, tokenizer):
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.sep_token_id == 3

    def test_encode_tags(self, tokenizer):
        ids = tokenizer.encode("<BOS><AMR_KANAMYCIN>")
        assert ids == [1, 5]

    def test_encode_dna(self, tokenizer):
        ids = tokenizer.encode("ATCG")
        assert len(ids) == 4

    def test_roundtrip(self, tokenizer):
        text = "<BOS><AMR_KANAMYCIN>ATCG"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_unknown_token(self, tokenizer):
        ids = tokenizer.encode("<UNKNOWN_TAG>")
        # Should map to UNK
        assert all(i == tokenizer._vocab.get("<UNK>", 4) for i in ids)

    def test_dna_bases_in_vocab(self, tokenizer):
        for base in "ATCGNatcgn":
            assert base in tokenizer.vocab
