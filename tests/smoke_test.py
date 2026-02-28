#!/usr/bin/env python3
"""Quick smoke test to verify basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_data_files():
    """Test that data files exist and are valid."""
    from pathlib import Path
    import pyarrow.parquet as pq
    
    data_dir = Path(__file__).parent.parent / "data"
    
    print("✓ Checking data files...")
    
    # Check special tokens
    special_tokens_path = data_dir / "special_tokens.txt"
    assert special_tokens_path.exists(), "Missing special_tokens.txt"
    with open(special_tokens_path) as f:
        tokens = [line.strip() for line in f if line.strip()]
    assert len(tokens) > 0, "special_tokens.txt is empty"
    print(f"  ✓ special_tokens.txt: {len(tokens)} tokens")
    
    # Check training pairs
    training_pairs_path = data_dir / "training_pairs_sample.parquet"
    assert training_pairs_path.exists(), "Missing training_pairs_sample.parquet"
    table = pq.read_table(training_pairs_path)
    assert "full_text" in table.column_names
    print(f"  ✓ training_pairs_sample.parquet: {len(table)} rows")
    
    # Check motif registry
    motif_path = data_dir / "motif_registry.parquet"
    assert motif_path.exists(), "Missing motif_registry.parquet"
    table = pq.read_table(motif_path)
    assert "token" in table.column_names
    assert "sequence" in table.column_names
    print(f"  ✓ motif_registry.parquet: {len(table)} rows")


def test_tokenizer():
    """Test that tokenizer can be created and works."""
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer
    import json
    import tempfile
    from pathlib import Path
    
    print("✓ Testing tokenizer...")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Build vocab from special tokens
    with open(data_dir / "special_tokens.txt") as f:
        special_tokens = [line.strip() for line in f if line.strip()]
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add DNA bases
    next_id = len(vocab)
    for base in "ATCGNatcgn":
        if base not in vocab:
            vocab[base] = next_id
            next_id += 1
    
    # Save vocab temporarily
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab, f)
        vocab_path = f.name
    
    try:
        tokenizer = PlasmidLMTokenizer(vocab_path)
        print(f"  ✓ Tokenizer created: {tokenizer.vocab_size} tokens")
        
        # Test encoding/decoding
        seq = "ATCGATCG"
        ids = tokenizer.encode(seq)
        decoded = tokenizer.decode(ids)
        assert decoded == seq, f"Roundtrip failed: {seq} != {decoded}"
        print(f"  ✓ DNA encoding/decoding works")
        
    finally:
        import os
        os.unlink(vocab_path)


def test_imports():
    """Test that key modules can be imported."""
    print("✓ Testing imports...")
    
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer
    print("  ✓ PlasmidLMTokenizer")
    
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMForCausalLM
    print("  ✓ PlasmidLMForCausalLM")
    
    from plasmid_llm.data import PlasmidDataset
    print("  ✓ PlasmidDataset")
    
    from plasmid_llm.config import PretrainingConfig
    print("  ✓ PretrainingConfig")

    from plasmid_llm.utils import load_config, setup_mlflow
    print("  ✓ plasmid_llm.utils")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("PlasmidLLM Smoke Tests")
    print("=" * 60)
    print()
    
    try:
        test_imports()
        print()
        test_data_files()
        print()
        test_tokenizer()
        print()
        print("=" * 60)
        print("✅ All smoke tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
