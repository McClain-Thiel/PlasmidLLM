"""Comprehensive integration tests with real data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
import pyarrow.parquet as pq
import torch

# Paths to data
DATA_DIR = Path(__file__).parent.parent / "data"
SPECIAL_TOKENS_PATH = DATA_DIR / "special_tokens.txt"
TRAINING_PAIRS_PATH = DATA_DIR / "training_pairs_sample.parquet"
MOTIF_LOOKUP_PATH = DATA_DIR / "motif_registry.parquet"


class TestDataFiles:
    """Test that data files exist and are valid."""
    
    def test_special_tokens_exists(self):
        """Test that special_tokens.txt exists."""
        assert SPECIAL_TOKENS_PATH.exists(), f"Missing: {SPECIAL_TOKENS_PATH}"
    
    def test_special_tokens_format(self):
        """Test that special_tokens.txt has correct format."""
        with open(SPECIAL_TOKENS_PATH) as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        assert len(tokens) > 0, "special_tokens.txt is empty"
        assert all(t.startswith("<") and t.endswith(">") for t in tokens), \
            "All tokens should be in <TOKEN> format"
        
        # Check for required tokens  
        required = ["<BOS>", "<EOS>", "<SEP>", "<PAD>", "<UNK>"]
        for req in required:
            assert req in tokens, f"Missing required token: {req}"
    
    def test_training_pairs_exists(self):
        """Test that training pairs parquet exists."""
        assert TRAINING_PAIRS_PATH.exists(), f"Missing: {TRAINING_PAIRS_PATH}"
    
    def test_training_pairs_structure(self):
        """Test that training pairs has expected columns."""
        table = pq.read_table(TRAINING_PAIRS_PATH)
        
        # Should have full_text column
        assert "full_text" in table.column_names, "Missing full_text column"
        assert len(table) > 0, "Empty training pairs file"
        
        # Check first row
        full_text = table.column("full_text")[0].as_py()
        assert isinstance(full_text, str), "full_text should be string"
        assert "<BOS>" in full_text, "full_text should contain <BOS>"
        # Note: Uses <SEQ> not <SEP> as separator
        assert "<SEQ>" in full_text or "<SEP>" in full_text, "full_text should contain separator token"
    
    def test_motif_lookup_exists(self):
        """Test that motif lookup parquet exists."""
        assert MOTIF_LOOKUP_PATH.exists(), f"Missing: {MOTIF_LOOKUP_PATH}"
    
    def test_motif_lookup_structure(self):
        """Test that motif lookup has expected columns."""
        table = pq.read_table(MOTIF_LOOKUP_PATH)
        
        required_cols = ["token", "sequence"]
        for col in required_cols:
            assert col in table.column_names, f"Missing column: {col}"
        
        assert len(table) > 0, "Empty motif lookup file"


class TestTokenizer:
    """Test tokenizer with real data."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer from real special tokens."""
        from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer
        
        # Build vocab from special tokens
        import json
        with open(SPECIAL_TOKENS_PATH) as f:
            special_tokens = [line.strip() for line in f if line.strip()]
        
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Add DNA bases
        next_id = len(vocab)
        for base in "ATCGNatcgn":
            if base not in vocab:
                vocab[base] = next_id
                next_id += 1
        
        # Save vocab temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vocab, f)
            vocab_path = f.name
        
        tokenizer = PlasmidLMTokenizer(vocab_path)
        
        # Cleanup
        import os
        os.unlink(vocab_path)
        
        return tokenizer
    
    def test_tokenizer_creation(self, tokenizer):
        """Test that tokenizer can be created."""
        assert tokenizer is not None
        assert tokenizer.vocab_size > 100  # Should have ~111 tokens
    
    def test_tokenizer_special_tokens(self, tokenizer):
        """Test that tokenizer has required special tokens."""
        vocab = tokenizer.get_vocab()
        
        required = ["<BOS>", "<EOS>", "<SEP>", "<PAD>", "<UNK>"]
        for token in required:
            assert token in vocab, f"Missing token: {token}"
    
    def test_tokenizer_dna_bases(self, tokenizer):
        """Test that tokenizer can encode DNA bases."""
        seq = "ATCGATCG"
        ids = tokenizer.encode(seq)
        
        assert len(ids) == 8
        assert all(isinstance(i, int) for i in ids)
        
        # Decode should recover original
        decoded = tokenizer.decode(ids)
        assert decoded == seq
    
    def test_tokenizer_full_text(self, tokenizer):
        """Test tokenizer on real training data."""
        table = pq.read_table(TRAINING_PAIRS_PATH, columns=["full_text"])
        full_text = table.column("full_text")[0].as_py()
        
        # Tokenize
        ids = tokenizer.encode(full_text)
        
        assert len(ids) > 0
        assert all(0 <= i < tokenizer.vocab_size for i in ids)
        
        # Decode - NOTE: This may not roundtrip perfectly if tokens in
        # training data aren't in vocab (like <SP_RAT>, <SEQ>)
        decoded = tokenizer.decode(ids)
        assert len(decoded) > 0
        assert "<BOS>" in decoded
        # If there are UNK tokens, it means some tokens in data aren't in vocab
        # This is expected for this test data
    
    def test_tokenizer_save_load(self, tokenizer, tmp_path):
        """Test save and load tokenizer."""
        save_dir = tmp_path / "tokenizer"
        
        # Save
        tokenizer.save_pretrained(save_dir)
        assert (save_dir / "vocab.json").exists()
        
        # Load
        from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer
        loaded = PlasmidLMTokenizer.from_pretrained(save_dir)
        
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.pad_token_id == tokenizer.pad_token_id


class TestDataset:
    """Test dataset loading with real data."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer fixture."""
        from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer
        import json
        import tempfile
        
        with open(SPECIAL_TOKENS_PATH) as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        vocab = {t: i for i, t in enumerate(tokens)}
        next_id = len(vocab)
        for base in "ATCGNatcgn":
            if base not in vocab:
                vocab[base] = next_id
                next_id += 1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vocab, f)
            vocab_path = f.name
        
        tok = PlasmidLMTokenizer(vocab_path)
        
        import os
        os.unlink(vocab_path)
        
        return tok
    
    def test_dataset_creation(self, tokenizer):
        """Test creating dataset from real data."""
        from plasmid_llm.data import PlasmidDataset
        
        dataset = PlasmidDataset(
            str(TRAINING_PAIRS_PATH),
            tokenizer,
            max_seq_len=512  # Smaller for testing
        )
        
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, tokenizer):
        """Test getting items from dataset."""
        from plasmid_llm.data import PlasmidDataset
        
        dataset = PlasmidDataset(
            str(TRAINING_PAIRS_PATH),
            tokenizer,
            max_seq_len=512
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        assert item["input_ids"].shape == (512,)
        assert item["attention_mask"].shape == (512,)
        assert item["labels"].shape == (512,)
        
        # Check types
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long
    
    def test_dataset_labels_masking(self, tokenizer):
        """Test that prompt is masked in labels."""
        from plasmid_llm.data import PlasmidDataset
        
        dataset = PlasmidDataset(
            str(TRAINING_PAIRS_PATH),
            tokenizer,
            max_seq_len=512
        )
        
        item = dataset[0]
        labels = item["labels"]
        
        # First tokens should be masked (prompt)
        assert labels[0].item() == -100  # BOS
        
        # Should have some -100 values (prompt masking)
        mask_count = (labels == -100).sum().item()
        assert mask_count > 0


parasail = pytest.importorskip("parasail")

@pytest.mark.skipif(
    not (DATA_DIR / "motif_registry.parquet").exists(),
    reason="Motif lookup required"
)
class TestRewardFunction:
    """Test reward function with real data."""
    
    def test_load_motif_lookup(self):
        """Test loading motif lookup."""
        from post_training.scorers.alignment import load_motif_lookup, AlignmentScorer
        
        lookup_df = load_motif_lookup(str(MOTIF_LOOKUP_PATH))
        
        assert len(lookup_df) > 0
        assert "token" in lookup_df.columns
        assert "sequence" in lookup_df.columns
    
    def test_reward_function_basic(self):
        """Test basic reward function."""
        from post_training.scorers.alignment import load_motif_lookup, AlignmentScorer

        lookup_df = load_motif_lookup(str(MOTIF_LOOKUP_PATH))
        scorer = AlignmentScorer(lookup_df=lookup_df)

        first_token = lookup_df.iloc[0]["token"]
        first_seq = lookup_df.iloc[0]["sequence"]
        prompt = f"<BOS>{first_token}<SEP>"

        reward_good = scorer.score_sequence(prompt, first_seq)
        reward_bad = scorer.score_sequence(prompt, "N" * 1000)

        assert reward_good > reward_bad


def test_config_classes():
    """Test config dataclasses."""
    from plasmid_llm.config import PretrainingConfig

    pretrain_config = PretrainingConfig(
        training_pairs=TRAINING_PAIRS_PATH,
        special_tokens=SPECIAL_TOKENS_PATH,
        hidden_size=128,
        num_hidden_layers=2,
    )

    assert pretrain_config.training_pairs.exists()
    assert pretrain_config.special_tokens.exists()
    assert pretrain_config.hidden_size == 128

    # Test to_mlflow_params
    params = pretrain_config.to_mlflow_params()
    assert "training_pairs" in params
    assert "training_pairs_hash" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
