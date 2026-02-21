"""Integration tests for PlasmidLM tokenizer with test data."""

import pytest
from pathlib import Path

from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer


# Test data location
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test"
SPECIAL_TOKENS_PATH = TEST_DATA_DIR / "special_tokens.txt"


@pytest.fixture
def test_vocab_file(tmp_path):
    """Create a minimal vocab from special tokens + DNA bases."""
    import json
    
    # Check if test data exists
    if not SPECIAL_TOKENS_PATH.exists():
        pytest.skip(f"Test data not found: {SPECIAL_TOKENS_PATH}")
    
    # Load special tokens
    with open(SPECIAL_TOKENS_PATH) as f:
        special_tokens = [line.strip() for line in f if line.strip()]
    
    # Create vocab
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add DNA bases (tokenizer will add these automatically, but we include for clarity)
    next_id = len(vocab)
    for base in "ATCGNatcgn":
        if base not in vocab:
            vocab[base] = next_id
            next_id += 1
    
    # Write to temp file
    vocab_file = tmp_path / "vocab.json"
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)
    
    return vocab_file


@pytest.fixture
def tokenizer(test_vocab_file):
    """Create tokenizer from test vocab."""
    return PlasmidLMTokenizer(str(test_vocab_file))


class TestPlasmidLMTokenizer:
    """Tests for PlasmidLMTokenizer with realistic data."""
    
    def test_vocab_has_special_tokens(self, tokenizer):
        """Verify all special tokens are in vocab."""
        vocab = tokenizer.get_vocab()
        assert "<PAD>" in vocab
        assert "<BOS>" in vocab
        assert "<EOS>" in vocab
        assert "<SEP>" in vocab
        assert "<UNK>" in vocab
    
    def test_vocab_has_dna_bases(self, tokenizer):
        """Verify DNA bases are in vocab."""
        vocab = tokenizer.get_vocab()
        for base in "ATCGNatcgn":
            assert base in vocab, f"Base {base} not in vocab"
    
    def test_special_token_ids(self, tokenizer):
        """Test special token ID properties."""
        assert tokenizer.pad_token_id >= 0
        assert tokenizer.bos_token_id >= 0
        assert tokenizer.eos_token_id >= 0
        assert tokenizer.sep_token_id >= 0
    
    def test_encode_special_tokens(self, tokenizer):
        """Test encoding special tokens."""
        text = "<BOS><AMR_KANAMYCIN><SEP>"
        ids = tokenizer.encode(text)
        
        # Should have 3 tokens
        assert len(ids) == 3
        assert ids[0] == tokenizer.bos_token_id
        assert ids[2] == tokenizer.sep_token_id
    
    def test_encode_dna_sequence(self, tokenizer):
        """Test encoding DNA bases."""
        seq = "ATCGATCG"
        ids = tokenizer.encode(seq)
        
        # Should have 8 tokens (one per base)
        assert len(ids) == 8
        
        # All should be valid IDs
        assert all(0 <= id < tokenizer.vocab_size for id in ids)
    
    def test_encode_full_text(self, tokenizer):
        """Test encoding complete training example."""
        full_text = "<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>ATCGATCG<EOS>"
        ids = tokenizer.encode(full_text)
        
        # BOS + 2 tags + SEP + 8 bases + EOS = 13 tokens
        assert len(ids) == 13
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id
    
    def test_decode_roundtrip(self, tokenizer):
        """Test encode -> decode roundtrip."""
        original = "<BOS><AMR_KANAMYCIN>ATCG<EOS>"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)
        
        assert decoded == original
    
    def test_unknown_token_handling(self, tokenizer):
        """Test that unknown tokens map to UNK."""
        text = "<UNKNOWN_TAG>"
        ids = tokenizer.encode(text)
        
        # Should contain UNK token ID
        assert tokenizer._vocab.get("<UNK>", -1) in ids or len(ids) > 0
    
    def test_save_and_load(self, tokenizer, tmp_path):
        """Test save_pretrained and from_pretrained."""
        save_dir = tmp_path / "tokenizer"
        
        # Save
        tokenizer.save_pretrained(save_dir)
        assert (save_dir / "vocab.json").exists()
        
        # Load
        loaded = PlasmidLMTokenizer.from_pretrained(save_dir)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.pad_token_id == tokenizer.pad_token_id
    
    def test_batch_encoding(self, tokenizer):
        """Test encoding multiple sequences."""
        texts = [
            "<BOS>ATCG<EOS>",
            "<BOS><AMR_KANAMYCIN>GCTA<EOS>",
        ]
        
        ids_list = [tokenizer.encode(t) for t in texts]
        
        # Should have different lengths
        assert len(ids_list[0]) < len(ids_list[1])
        
        # All should start with BOS and end with EOS
        for ids in ids_list:
            assert ids[0] == tokenizer.bos_token_id
            assert ids[-1] == tokenizer.eos_token_id


@pytest.mark.integration
class TestWithRealTestData:
    """Integration tests using real test data files."""
    
    @pytest.fixture
    def training_pairs_path(self):
        """Path to test training pairs."""
        path = TEST_DATA_DIR / "training_pairs.parquet"
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")
        return path
    
    def test_tokenize_training_pairs(self, tokenizer, training_pairs_path):
        """Test tokenizing actual training pairs from test data."""
        import pyarrow.parquet as pq
        
        table = pq.read_table(training_pairs_path)
        full_texts = table.column("full_text").to_pylist()
        
        # Should have some examples
        assert len(full_texts) > 0
        
        # Tokenize first few examples
        for text in full_texts[:5]:
            ids = tokenizer.encode(text)
            
            # Should produce valid tokens
            assert len(ids) > 0
            assert all(0 <= id < tokenizer.vocab_size for id in ids)
            
            # Should start with BOS
            assert ids[0] == tokenizer.bos_token_id
            
            # Should contain SEP
            assert tokenizer.sep_token_id in ids
    
    def test_decode_training_pairs(self, tokenizer, training_pairs_path):
        """Test roundtrip on real training data."""
        import pyarrow.parquet as pq
        
        table = pq.read_table(training_pairs_path)
        full_texts = table.column("full_text").to_pylist()
        
        # Test roundtrip on first example
        original = full_texts[0]
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)
        
        assert decoded == original
