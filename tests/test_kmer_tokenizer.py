"""Tests for k-mer tokenizer."""

import json
import tempfile
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plasmid_llm.models.hf_plasmid_lm.tokenization_kmer import (
    PlasmidKmerTokenizer,
    build_kmer_vocab,
)


SPECIAL_TOKENS = [
    "<BOS>", "<EOS>", "<SEP>", "<PAD>", "<UNK>", "<SEQ>",
    "<AMR_AMPICILLIN>", "<ORI_COLE1>", "<PROM_CMV>",
]


@pytest.fixture
def vocab_file(tmp_path):
    vocab = build_kmer_vocab(SPECIAL_TOKENS, k=6)
    vf = tmp_path / "vocab.json"
    with open(vf, "w") as f:
        json.dump(vocab, f)
    return str(vf)


@pytest.fixture
def tokenizer_s3(vocab_file):
    return PlasmidKmerTokenizer(vocab_file, k=6, stride=3)


@pytest.fixture
def tokenizer_s4(vocab_file):
    return PlasmidKmerTokenizer(vocab_file, k=6, stride=4)


class TestBuildVocab:
    def test_vocab_size(self):
        vocab = build_kmer_vocab(SPECIAL_TOKENS, k=6)
        assert len(vocab) == len(SPECIAL_TOKENS) + 4**6  # 9 + 4096 = 4105

    def test_special_tokens_first(self):
        vocab = build_kmer_vocab(SPECIAL_TOKENS, k=6)
        for i, tok in enumerate(SPECIAL_TOKENS):
            assert vocab[tok] == i

    def test_all_kmers_present(self):
        vocab = build_kmer_vocab(SPECIAL_TOKENS, k=6)
        assert "AAAAAA" in vocab
        assert "TTTTTT" in vocab
        assert "ACGTAC" in vocab
        assert "CCCCCC" in vocab


class TestCleanDNA:
    def test_uppercase(self, tokenizer_s3):
        assert tokenizer_s3._clean_dna("atcg") == "ATCG"

    def test_strip_whitespace(self, tokenizer_s3):
        assert tokenizer_s3._clean_dna("AT CG\nTT") == "ATCGTT"

    def test_n_replacement_deterministic(self, tokenizer_s3):
        result1 = tokenizer_s3._clean_dna("ANNCG")
        result2 = tokenizer_s3._clean_dna("ANNCG")
        assert result1 == result2
        assert "N" not in result1

    def test_n_all_replaced(self, tokenizer_s3):
        result = tokenizer_s3._clean_dna("NNNNNNNNN")
        assert "N" not in result
        assert len(result) == 9
        assert all(c in "ACGT" for c in result)


class TestKmerize:
    def test_exact_kmer(self, tokenizer_s3):
        """6 bases = exactly 1 k-mer."""
        result = tokenizer_s3._kmerize("ACGTAC")
        assert result == ["ACGTAC"]

    def test_stride3_overlap(self, tokenizer_s3):
        """12 bases with stride=3 => positions 0,3,6 => 3 k-mers."""
        dna = "ACGTACGTACGT"
        result = tokenizer_s3._kmerize(dna)
        assert result == ["ACGTAC", "TACGTA", "GTACGT"]

    def test_stride4_overlap(self, tokenizer_s4):
        """12 bases with stride=4 => positions 0,4 => 2 k-mers, with tail."""
        dna = "ACGTACGTACGT"
        result = tokenizer_s4._kmerize(dna)
        # positions 0,4 => ACGTAC, ACGTAC then check tail
        assert len(result) >= 2
        # First k-mer
        assert result[0] == "ACGTAC"

    def test_short_sequence(self, tokenizer_s3):
        """Sequence shorter than k gets padded."""
        result = tokenizer_s3._kmerize("ACG")
        assert result == ["ACGAAA"]  # padded with A

    def test_empty(self, tokenizer_s3):
        assert tokenizer_s3._kmerize("") == []


class TestTokenizeAndDecode:
    def test_roundtrip_pure_dna(self, tokenizer_s3):
        """Encode then decode should recover the original DNA (up to padding at tail)."""
        dna = "ACGTACGTACGTACGTACGT"  # 20 bases
        tokens = tokenizer_s3._tokenize(dna)
        decoded = tokenizer_s3.convert_tokens_to_string(tokens)
        # The decoded DNA should start with the original
        assert decoded.startswith(dna[:len(dna) - (len(dna) % tokenizer_s3.stride) + tokenizer_s3.stride][:len(dna)]
                                  if len(dna) >= tokenizer_s3.k else dna)

    def test_roundtrip_exact_multiple(self, tokenizer_s3):
        """When length is k + n*stride, roundtrip should be exact."""
        # k=6, stride=3: length = 6 + 2*3 = 12
        dna = "ACGTACGTACGT"
        tokens = tokenizer_s3._tokenize(dna)
        decoded = tokenizer_s3.convert_tokens_to_string(tokens)
        assert decoded == dna

    def test_special_tokens_preserved(self, tokenizer_s3):
        text = "<BOS><AMR_AMPICILLIN><SEP>ACGTACGTACGT<EOS>"
        tokens = tokenizer_s3._tokenize(text)
        assert tokens[0] == "<BOS>"
        assert tokens[1] == "<AMR_AMPICILLIN>"
        assert tokens[2] == "<SEP>"
        assert tokens[-1] == "<EOS>"
        # DNA k-mers in the middle
        dna_tokens = [t for t in tokens if not t.startswith("<")]
        assert len(dna_tokens) > 0

    def test_encode_decode_ids(self, tokenizer_s3):
        """Full encode/decode roundtrip via token IDs."""
        text = "<BOS><ORI_COLE1><SEP>ACGTACGTACGT<EOS>"
        ids = tokenizer_s3.encode(text, add_special_tokens=False)
        decoded = tokenizer_s3.decode(ids)
        assert "<BOS>" in decoded
        assert "<ORI_COLE1>" in decoded
        assert "ACGTACGTACGT" in decoded

    def test_n_handling_in_full_pipeline(self, tokenizer_s3):
        """N bases should be replaced and result in valid k-mer tokens."""
        text = "ACGTNACGTNA"
        tokens = tokenizer_s3._tokenize(text)
        for t in tokens:
            assert "N" not in t

    def test_vocab_size(self, tokenizer_s3):
        assert tokenizer_s3.vocab_size == len(SPECIAL_TOKENS) + 4**6

    def test_special_token_ids(self, tokenizer_s3):
        assert tokenizer_s3.bos_token_id == 0
        assert tokenizer_s3.eos_token_id == 1
        assert tokenizer_s3.sep_token_id == 2
        assert tokenizer_s3.pad_token_id == 3


class TestSaveLoad:
    def test_save_and_reload(self, tokenizer_s3, tmp_path):
        save_dir = str(tmp_path / "saved")
        tokenizer_s3.save_pretrained(save_dir)
        # Reload — need to pass k and stride
        loaded = PlasmidKmerTokenizer.from_pretrained(save_dir, k=6, stride=3)
        assert loaded.vocab_size == tokenizer_s3.vocab_size
        # Test encoding is identical
        text = "ACGTACGTACGT"
        assert tokenizer_s3.encode(text) == loaded.encode(text)
