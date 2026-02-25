"""Tests for the reward function."""

import pytest
from pathlib import Path

# Skip if dependencies not installed
pytest.importorskip("parasail")
pytest.importorskip("Bio")
pytest.importorskip("pandas")

from post_training.reward import (
    _extract_category,
    build_category_index,
    compute_reward,
    parse_hard_tokens,
    plasmid_reward_fn,
    safe_translate,
)


class TestRewardFunction:
    """Basic tests for reward function logic."""
    
    def test_safe_translate(self):
        """Test DNA translation."""
        # Valid DNA sequence (3 codons)
        dna = "ATGATGCCC"
        protein = safe_translate(dna)
        assert protein is not None
        assert len(protein) == 3
        
        # Too short
        assert safe_translate("AT") is None
        assert safe_translate("") is None
        
        # Non-multiple of 3 (should trim)
        dna = "ATGATGCCCA"  # 10 bases
        protein = safe_translate(dna)
        assert protein is not None
        assert len(protein) == 3  # Only first 9 bases translated
    
    def test_parse_hard_tokens_basic(self):
        """Test hard token parsing without lookup."""
        import pandas as pd
        
        # Create minimal lookup
        lookup_df = pd.DataFrame({
            "token": ["<AMR_KANAMYCIN>", "<ORI_COLE1>", "<PROM_T7>"],
        }).set_index("token", drop=False)
        
        # Parse various prompts
        prompt1 = "<BOS><AMR_KANAMYCIN><SEP>"
        tokens1 = parse_hard_tokens(prompt1, lookup_df)
        assert "<AMR_KANAMYCIN>" in tokens1
        
        prompt2 = "<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>"
        tokens2 = parse_hard_tokens(prompt2, lookup_df)
        assert len(tokens2) == 2
        assert "<AMR_KANAMYCIN>" in tokens2
        assert "<ORI_COLE1>" in tokens2
        
        # Unknown token should be filtered out
        prompt3 = "<BOS><UNKNOWN_TOKEN><SEP>"
        tokens3 = parse_hard_tokens(prompt3, lookup_df)
        assert len(tokens3) == 0
    
    def test_plasmid_reward_fn_batch(self):
        """Test batch reward function API."""
        import pandas as pd
        
        # Minimal lookup
        lookup_df = pd.DataFrame({
            "token": ["<AMR_KANAMYCIN>"],
            "dna_seq": ["ATGATG" * 100],  # Dummy sequence
            "is_cds": [True],
            "seq_type": ["dna"],
            "dna_max_score": [100],
            "protein_max_score": [100],
        }).set_index("token", drop=False)
        
        prompts = [
            "<BOS><AMR_KANAMYCIN><SEP>",
            "<BOS><AMR_KANAMYCIN><SEP>",
        ]
        
        # Random DNA sequences
        completions = [
            "ATGATGATG" * 50,
            "GCGCGCGCG" * 50,
        ]
        
        rewards = plasmid_reward_fn(prompts, completions, lookup_df)
        
        assert len(rewards) == 2
        assert all(0.0 <= r <= 1.0 for r in rewards)
        
        # First completion has matching sequence, should score higher
        assert rewards[0] > rewards[1]
    
    def test_short_sequence_penalty(self):
        """Test that short sequences get 0 reward."""
        import pandas as pd
        
        lookup_df = pd.DataFrame({
            "token": ["<AMR_KANAMYCIN>"],
            "dna_seq": ["ATGATG" * 100],
            "is_cds": [True],
            "seq_type": ["dna"],
            "dna_max_score": [100],
            "protein_max_score": [100],
        }).set_index("token", drop=False)
        
        prompts = ["<BOS><AMR_KANAMYCIN><SEP>"]
        short_completion = ["ATGATG"]  # Too short (< 100 bp)
        
        rewards = plasmid_reward_fn(prompts, short_completion, lookup_df)
        assert rewards[0] == 0.0


class TestCurriculumReward:
    """Tests for alpha-blended curriculum reward scheduling using real plasmid data."""

    REGISTRY_PATH = Path("data/motif_registry.parquet")
    PAIRS_PATH = Path("data/training_pairs_sample.parquet")

    @pytest.fixture(scope="class")
    def real_data(self):
        """Load motif registry and training pairs; skip if files unavailable."""
        import pandas as pd
        from post_training.reward import load_motif_lookup

        if not self.REGISTRY_PATH.exists() or not self.PAIRS_PATH.exists():
            pytest.skip("Real data files not available (motif_registry / training_pairs_sample)")

        lookup_df = load_motif_lookup(str(self.REGISTRY_PATH))
        pairs = pd.read_parquet(self.PAIRS_PATH)
        category_index = build_category_index(lookup_df)
        return lookup_df, pairs, category_index

    @staticmethod
    def _pick_pair(pairs, token_must_contain: str):
        """Find the shortest training pair whose prompt contains a specific token."""
        mask = pairs["prompt"].str.contains(token_must_contain, regex=False)
        subset = pairs[mask].sort_values("sequence_length")
        assert len(subset) > 0, f"No rows with {token_must_contain}"
        row = subset.iloc[0]
        return row["prompt"], row["sequence"]

    def test_alpha_1_matches_original(self, real_data):
        """alpha=1.0 with category_index must reproduce no-alpha behavior exactly."""
        lookup_df, pairs, category_index = real_data
        prompt, seq = self._pick_pair(pairs, "<AMR_KANAMYCIN>")

        reward_original = compute_reward(prompt, seq, lookup_df)
        reward_alpha1 = compute_reward(
            prompt, seq, lookup_df, alpha=1.0, category_index=category_index,
        )

        assert reward_original == reward_alpha1

    def test_alpha_1_no_category_index(self, real_data):
        """alpha=1.0 with category_index=None must also match default behavior."""
        lookup_df, pairs, _ = real_data
        prompt, seq = self._pick_pair(pairs, "<AMR_KANAMYCIN>")

        reward_default = compute_reward(prompt, seq, lookup_df)
        reward_none = compute_reward(
            prompt, seq, lookup_df, alpha=1.0, category_index=None,
        )

        assert reward_default == reward_none

    def test_alpha_0_rewards_category_presence(self, real_data):
        """alpha=0.0 should reward a real plasmid whose AMR is the wrong subtype.

        Uses a kanamycin prompt evaluated against an ampicillin plasmid's sequence.
        The ampicillin sequence genuinely contains an AMR gene, just not the one
        the prompt asked for — presence scoring should give credit.
        """
        lookup_df, pairs, category_index = real_data
        kan_prompt, _ = self._pick_pair(pairs, "<AMR_KANAMYCIN>")
        _, amp_seq = self._pick_pair(pairs, "<AMR_AMPICILLIN>")

        # Evaluate amp sequence against kan prompt
        reward_exact = compute_reward(
            kan_prompt, amp_seq, lookup_df, alpha=1.0, category_index=category_index,
        )
        reward_presence = compute_reward(
            kan_prompt, amp_seq, lookup_df, alpha=0.0, category_index=category_index,
        )

        assert reward_presence > 0.0, "Presence reward should be positive for real sequence"
        assert reward_presence > reward_exact, (
            f"Presence reward ({reward_presence:.4f}) should exceed exact-match "
            f"reward ({reward_exact:.4f}) when AMR subtype is mismatched"
        )

    def test_alpha_blend_interpolates(self, real_data):
        """alpha=0.5 should produce a reward between alpha=0 and alpha=1."""
        lookup_df, pairs, category_index = real_data
        kan_prompt, _ = self._pick_pair(pairs, "<AMR_KANAMYCIN>")
        _, amp_seq = self._pick_pair(pairs, "<AMR_AMPICILLIN>")

        reward_0 = compute_reward(
            kan_prompt, amp_seq, lookup_df, alpha=0.0, category_index=category_index,
        )
        reward_1 = compute_reward(
            kan_prompt, amp_seq, lookup_df, alpha=1.0, category_index=category_index,
        )
        reward_mid = compute_reward(
            kan_prompt, amp_seq, lookup_df, alpha=0.5, category_index=category_index,
        )

        low, high = min(reward_0, reward_1), max(reward_0, reward_1)
        assert low <= reward_mid <= high, (
            f"Blended reward {reward_mid:.4f} outside [{low:.4f}, {high:.4f}]"
        )

    def test_plasmid_reward_fn_passes_alpha(self, real_data):
        """Batch API should pass alpha through and produce different rewards."""
        lookup_df, pairs, category_index = real_data
        kan_prompt, _ = self._pick_pair(pairs, "<AMR_KANAMYCIN>")
        _, amp_seq = self._pick_pair(pairs, "<AMR_AMPICILLIN>")

        prompts = [kan_prompt]
        completions = [amp_seq]

        rewards_exact = plasmid_reward_fn(
            prompts, completions, lookup_df, alpha=1.0, category_index=category_index,
        )
        rewards_presence = plasmid_reward_fn(
            prompts, completions, lookup_df, alpha=0.0, category_index=category_index,
        )

        assert rewards_presence[0] > rewards_exact[0], (
            f"Presence reward ({rewards_presence[0]:.4f}) should exceed exact "
            f"({rewards_exact[0]:.4f}) via batch API"
        )


@pytest.mark.integration
class TestWithMotifLookup:
    """Integration tests with actual motif lookup file (if available)."""
    
    @pytest.fixture
    def motif_lookup_path(self):
        """Path to motif lookup parquet."""
        # Try common locations
        candidates = [
            Path("data/motif_lookup.parquet"),
            Path("data/test/motif_lookup.parquet"),
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        pytest.skip("Motif lookup file not found")
    
    def test_load_motif_lookup(self, motif_lookup_path):
        """Test loading real motif lookup."""
        from post_training.reward import load_motif_lookup
        
        df = load_motif_lookup(str(motif_lookup_path))
        
        # Check expected columns
        assert "token" in df.columns
        assert "dna_seq" in df.columns
        assert "dna_max_score" in df.columns
        
        # Should have some tokens
        assert len(df) > 0
        
        # Max scores should be computed
        assert df["dna_max_score"].notna().any()
    
    def test_reward_with_real_lookup(self, motif_lookup_path):
        """Test reward computation with real data."""
        from post_training.reward import load_motif_lookup, compute_reward
        
        lookup_df = load_motif_lookup(str(motif_lookup_path))
        
        # Get first token
        first_token = lookup_df.index[0]
        prompt = f"<BOS>{first_token}<SEP>"
        
        # Get actual sequence for that token
        token_data = lookup_df.loc[first_token]
        if isinstance(token_data, pd.Series):
            actual_seq = token_data["dna_seq"]
        else:
            actual_seq = token_data.iloc[0]["dna_seq"]
        
        # Compute reward with actual sequence (should be high)
        reward_good = compute_reward(prompt, actual_seq, lookup_df)
        
        # Compute reward with random sequence (should be low)
        random_seq = "N" * len(actual_seq)
        reward_bad = compute_reward(prompt, random_seq, lookup_df)
        
        # Actual sequence should score much higher
        assert reward_good > reward_bad
        assert reward_good > 0.5  # Should pass QC threshold
