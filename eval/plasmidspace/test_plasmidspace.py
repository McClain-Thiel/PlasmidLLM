"""Tests for PlasmidSpace app components.

Run with: python -m pytest test_plasmidspace.py -v
Requires the model and plannotate to be installed.
"""

import re
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_ID = "McClain/PlasmidLM-kmer6-GRPO-plannotate"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="session")
def model():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
    )
    m.eval()
    return m


# Sample prompts from training_pairs_v4.parquet
SAMPLE_PROMPTS = [
    "<BOS><VEC_BACTERIAL><AMR_KANAMYCIN><SEQ>",
    "<BOS><VEC_BACTERIAL><SP_HUMAN><COPY_HIGH><AMR_KANAMYCIN><ORI_COLE1><PROM_LAC><ELEM_IRES><ELEM_TRACRRNA><SEQ>",
    "<BOS><VEC_INSECT><SP_DROSOPHILA><AMR_AMPICILLIN><ORI_COLE1><PROM_AMPR><PROM_LAC><ELEM_IRES><ELEM_TRACRRNA><TAG_FLAG><TAG_V5><SEQ>",
]

# A real DNA sequence from training data (first 500bp of a bacterial plasmid)
SAMPLE_DNA = (
    "ATCACCACCCATACCCATATTCTTCTTCCCACTATCTTTACCACTCACAGAAGCTAAAATCGATTCGATATTAAACGACGCCGAACTCATCTCTCCATCC"
    "ATTTACGCCGGGATCAGGGTCCTGATAGCGCATGACTTCTAATCATATGTACCCTCTCCTAAAGTTGATCTATTGTGGCATTTCTACTAATCATAAAATC"
    "TCTCAAGGAGAAATCATTTCGGCCACTATCCTGCTAAAAAGCCCGACTCCCGTTATCCTCCTGCGGATCACTTTTAAAATGACCCGACGTCACGCCTCTC"
    "TTAATCGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAA"
    "TCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGCCGCGTTGCTGGCGTTTTTCCATAGGCTCCGCC"
    "CCCCTGACGA"
)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_special_tokens_exist(self, tokenizer):
        vocab = tokenizer.get_vocab()
        for tok in ["<BOS>", "<EOS>", "<SEQ>", "<PAD>"]:
            assert tok in vocab, f"Missing special token: {tok}"

    def test_functional_tokens_exist(self, tokenizer):
        vocab = tokenizer.get_vocab()
        required = [
            "<AMR_KANAMYCIN>", "<AMR_AMPICILLIN>", "<ORI_COLE1>",
            "<PROM_T7>", "<VEC_BACTERIAL>", "<VEC_LENTIVIRAL>",
            "<REPORTER_GFP>", "<ELEM_IRES>",
        ]
        for tok in required:
            assert tok in vocab, f"Missing functional token: {tok}"

    def test_prompt_tokenization_roundtrip(self, tokenizer):
        for prompt in SAMPLE_PROMPTS:
            ids = tokenizer(prompt)["input_ids"]
            decoded = tokenizer.decode(ids)
            # Decoded should contain same tokens (whitespace may differ)
            for tok in re.findall(r"<[^>]+>", prompt):
                assert tok in decoded, f"{tok} lost in roundtrip for: {prompt}"

    def test_sep_vs_seq(self, tokenizer):
        """<SEQ> (id=5) is the correct separator, not <SEP> (id=2)."""
        vocab = tokenizer.get_vocab()
        assert vocab["<SEQ>"] == 5
        assert vocab["<SEP>"] == 2
        # Training data uses <SEQ>
        ids = tokenizer("<BOS><AMR_KANAMYCIN><SEQ>")["input_ids"]
        assert ids[-1] == 5, "Last token should be <SEQ> (id=5)"

    def test_spaces_dont_matter(self, tokenizer):
        ids_nospace = tokenizer("<BOS><AMR_KANAMYCIN><ORI_COLE1><SEQ>")["input_ids"]
        ids_space = tokenizer("<BOS> <AMR_KANAMYCIN> <ORI_COLE1> <SEQ>")["input_ids"]
        assert ids_nospace == ids_space


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------


class TestGeneration:
    def test_generates_dna(self, model, tokenizer):
        """Model generates valid DNA characters after a prompt."""
        prompt = SAMPLE_PROMPTS[0]
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=100,
                temperature=0.3, do_sample=True, top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=False)
        # Extract DNA portion (after <SEQ>)
        dna_part = generated.split("<SEQ>")[-1]
        dna_clean = re.sub(r"<[^>]+>", "", dna_part.upper())
        dna_clean = re.sub(r"[^ATGCN]", "", dna_clean)
        assert len(dna_clean) > 0, "No DNA generated"
        assert set(dna_clean) <= {"A", "T", "G", "C", "N"}, (
            f"Invalid DNA characters: {set(dna_clean) - {'A', 'T', 'G', 'C', 'N'}}"
        )

    def test_different_prompts_different_output(self, model, tokenizer):
        """Different prompts should produce different sequences."""
        dnas = []
        for prompt in SAMPLE_PROMPTS[:2]:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=50,
                    temperature=0.01, do_sample=True, top_k=5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(output[0])
            dna = re.sub(r"<[^>]+>", "", raw.split("<SEQ>")[-1].upper())
            dna = re.sub(r"[^ATGCN]", "", dna)
            dnas.append(dna)
        assert dnas[0] != dnas[1], "Different prompts produced identical output"

    def test_generates_reasonable_length(self, model, tokenizer):
        """With 500 max tokens, should generate substantial DNA."""
        prompt = SAMPLE_PROMPTS[1]
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=500,
                temperature=0.3, do_sample=True, top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output[0])
        dna = re.sub(r"<[^>]+>", "", raw.split("<SEQ>")[-1].upper())
        dna = re.sub(r"[^ATGCN]", "", dna)
        # kmer6 with stride 3: ~500 tokens * 3bp = ~1500bp
        assert len(dna) >= 500, f"Expected >= 500bp, got {len(dna)}bp"


# ---------------------------------------------------------------------------
# pLannotate annotation tests
# ---------------------------------------------------------------------------


class TestAnnotation:
    def test_plannotate_import(self):
        from plannotate.annotate import annotate
        from plannotate.bokeh_plot import get_bokeh
        assert callable(annotate)
        assert callable(get_bokeh)

    def test_annotate_real_sequence(self):
        """Annotate a real plasmid fragment — should find features."""
        from plannotate.annotate import annotate
        # Use a longer sequence for better chance of hits
        hits = annotate(SAMPLE_DNA, is_detailed=True, linear=False)
        # Should return a DataFrame (may or may not have hits on this fragment)
        assert hits is not None
        assert hasattr(hits, "columns")

    def test_bokeh_plot_renders(self):
        """If annotations found, bokeh plot should produce HTML."""
        from plannotate.annotate import annotate
        from plannotate.bokeh_plot import get_bokeh
        from bokeh.embed import file_html
        from bokeh.resources import CDN

        hits = annotate(SAMPLE_DNA, is_detailed=True, linear=False)
        if hits is not None and not hits.empty:
            fig = get_bokeh(hits, linear=False)
            html = file_html(fig, CDN, "Test")
            assert "<html" in html.lower() or "<div" in html.lower()


# ---------------------------------------------------------------------------
# App helper tests (no model needed)
# ---------------------------------------------------------------------------


class TestAppHelpers:
    def test_ensure_prompt_format(self):
        from app import _ensure_prompt_format
        assert _ensure_prompt_format("<AMR_KANAMYCIN>") == "<BOS> <AMR_KANAMYCIN> <SEQ>"
        assert _ensure_prompt_format("<BOS> <AMR_KANAMYCIN> <SEQ>") == "<BOS> <AMR_KANAMYCIN> <SEQ>"

    def test_clean_dna(self):
        from app import _clean_dna
        assert _clean_dna("ATGC<EOS>") == "ATGC"
        assert _clean_dna("atgcNNN") == "ATGCNNN"
        assert _clean_dna("<some_token>AATTCC<EOS>") == "AATTCC"
        assert _clean_dna("") == ""

    def test_token_categories_populated(self):
        from app import TOKEN_BY_CATEGORY, SPECIAL_TOKENS
        assert len(SPECIAL_TOKENS) > 50, f"Only {len(SPECIAL_TOKENS)} special tokens"
        assert "AMR" in TOKEN_BY_CATEGORY
        assert "VEC" in TOKEN_BY_CATEGORY
        assert "ORI" in TOKEN_BY_CATEGORY
        assert "PROM" in TOKEN_BY_CATEGORY
