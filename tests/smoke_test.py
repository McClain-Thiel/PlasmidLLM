#!/usr/bin/env python3
"""Quick smoke test to verify basic functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_data_files():
    """Test that tracked vocabulary files exist and local parquets have expected schemas."""
    import json
    from pathlib import Path
    import pyarrow.parquet as pq

    data_dir = Path(__file__).parent.parent / "data"

    print("✓ Checking data files...")

    # Check special tokens (tracked in Git)
    special_tokens_path = data_dir / "special_tokens.txt"
    assert special_tokens_path.exists(), "Missing special_tokens.txt"
    with open(special_tokens_path) as f:
        tokens = [line.strip() for line in f if line.strip()]
    assert len(tokens) == 106, f"Expected 106 special tokens, got {len(tokens)}"
    print(f"  ✓ special_tokens.txt: {len(tokens)} tokens")

    # Check token vocabulary mapping (tracked in Git)
    vocab_path = data_dir / "token_vocabulary_v4.json"
    assert vocab_path.exists(), "Missing token_vocabulary_v4.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    assert len(vocab) == 106, f"Expected 106 vocabulary entries, got {len(vocab)}"
    print(f"  ✓ token_vocabulary_v4.json: {len(vocab)} entries")

    # Check external training/validation parquet when downloaded locally
    training_candidates = [
        data_dir / "cluster_v3" / "validation.parquet",
        data_dir / "training_pairs_v4.parquet",
    ]
    training_pairs_path = next((p for p in training_candidates if p.exists()), None)
    if training_pairs_path is not None:
        table = pq.read_table(training_pairs_path)
        assert "full_text" in table.column_names
        print(f"  ✓ {training_pairs_path.relative_to(data_dir)}: {len(table)} rows")

    # Check external motif registry parquet when downloaded locally
    motif_candidates = [
        data_dir / "motif_registry_combined.parquet",
        data_dir / "motif_registry.parquet",
    ]
    motif_path = next((p for p in motif_candidates if p.exists()), None)
    if motif_path is not None:
        table = pq.read_table(motif_path)
        assert "token" in table.column_names
        assert "sequence" in table.column_names
        print(f"  ✓ {motif_path.relative_to(data_dir)}: {len(table)} rows")


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
        print("  ✓ DNA encoding/decoding works")
        
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
