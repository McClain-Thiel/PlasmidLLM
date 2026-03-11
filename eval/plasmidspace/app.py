"""PlasmidSpace – AI-powered plasmid design demo.

Generates synthetic plasmid DNA from natural language descriptions using
PlasmidLM, then annotates with pLannotate for visualization.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# numpy compat — bokeh 2.4.3 uses numpy.bool8, removed in numpy 2.x.
# ---------------------------------------------------------------------------
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ---------------------------------------------------------------------------
# Streamlit shim — plannotate imports streamlit at the top level.
# ---------------------------------------------------------------------------
import sys
import types

if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")

    class _Noop:
        def __call__(self, *a, **kw):
            return self
        def __getattr__(self, name):
            return self

    _st.cache = lambda func, *a, **kw: func
    _st.progress = lambda n: _Noop()
    _st.error = lambda msg: None
    sys.modules["streamlit"] = _st
    sys.modules["streamlit.cli"] = types.ModuleType("streamlit.cli")

# ---------------------------------------------------------------------------

import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import gradio as gr
import pandas as pd
import torch
from bokeh.embed import file_html
from bokeh.resources import CDN as BOKEH_CDN
from plannotate.annotate import annotate as _plannotate_annotate
from plannotate.bokeh_plot import get_bokeh as _plannotate_bokeh
from plannotate import resources as _plannotate_rsc
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "McClain/PlasmidLM-kmer6-GRPO-plannotate"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FUNCTIONAL_PREFIXES = frozenset(
    {"AMR_", "ORI_", "PROM_", "REPORTER_", "TAG_", "ELEM_",
     "BB_", "VEC_", "SP_", "COPY_"}
)

# Prefixes for "hard" tokens that plannotate can actually verify
HARD_PREFIXES = frozenset(
    {"AMR_", "ORI_", "PROM_", "REPORTER_", "TAG_", "ELEM_"}
)

# ---------------------------------------------------------------------------
# plannotate Feature name → PlasmidLM token mapping
# ---------------------------------------------------------------------------
# Maps plannotate annotation Feature names (from SnapGene/fpbase DB) to
# the inner part of PlasmidLM tokens (e.g. "KanR" → "AMR_KANAMYCIN").
# Used to score how well a generated sequence matches the requested tokens.

_FEATURE_TO_TOKEN: dict[str, str] = {
    # --- Antibiotic resistance ---
    "KanR": "AMR_KANAMYCIN",
    "kanMX": "AMR_KANAMYCIN",
    "NeoR/KanR": "AMR_KANAMYCIN",
    "AmpR": "AMR_AMPICILLIN",
    "bla": "AMR_AMPICILLIN",
    "PuroR": "AMR_PUROMYCIN",
    "HygR": "AMR_HYGROMYCIN",
    "hph": "AMR_HYGROMYCIN",
    "hphMX6": "AMR_HYGROMYCIN",
    "BleoR": "AMR_BLEOMYCIN",
    "CmR": "AMR_CHLORAMPHENICOL",
    "cat": "AMR_CHLORAMPHENICOL",
    "TetR": "AMR_TETRACYCLINE",
    "SmR": "AMR_SPECTINOMYCIN",
    "SpecR": "AMR_SPECTINOMYCIN",
    "GentR": "AMR_GENTAMICIN",
    "ZeoR": "AMR_ZEOCIN",
    "NatR": "AMR_NOURSEOTHRICIN",
    "BsdR": "AMR_BLASTICIDIN",
    # --- Origins of replication ---
    "ori": "ORI_COLE1",
    "ColE1 origin": "ORI_COLE1",
    "pBR322 ori": "ORI_COLE1",
    "pMB1 ori": "ORI_COLE1",
    "pUC ori": "ORI_COLE1",
    "p15A ori": "ORI_P15A",
    "pSC101 ori": "ORI_PSC101",
    "f1 ori": "ORI_F1",
    "M13 ori": "ORI_M13",
    "SV40 ori": "ORI_SV40",
    "2μ ori": "ORI_2MICRON",
    "2u ori": "ORI_2MICRON",
    "CEN/ARS": "ORI_CENARS",
    # --- Promoters ---
    "T7 promoter": "PROM_T7",
    "T7promoter": "PROM_T7",
    "lac promoter": "PROM_LAC",
    "lac operator": "PROM_LAC",
    "CMV promoter": "PROM_CMV",
    "CMV enhancer + promoter": "PROM_CMV",
    "SV40 promoter": "PROM_SV40",
    "SV40 early promoter": "PROM_SV40",
    "U6 promoter": "PROM_U6",
    "AmpR promoter": "PROM_AMPR",
    "CAG promoter": "PROM_CAG",
    "EF-1α promoter": "PROM_EF1A",
    "EF1a promoter": "PROM_EF1A",
    "PGK promoter": "PROM_PGK",
    "UBC promoter": "PROM_UBC",
    "hU6 promoter": "PROM_U6",
    "T3 promoter": "PROM_T3",
    "SP6 promoter": "PROM_SP6",
    "trc promoter": "PROM_TRC",
    "tac promoter": "PROM_TAC",
    # --- Tags ---
    "6xHis": "TAG_HIS",
    "8xHis": "TAG_HIS",
    "10xHis": "TAG_HIS",
    "His-tag": "TAG_HIS",
    "FLAG": "TAG_FLAG",
    "2xFLAG": "TAG_FLAG",
    "3xFLAG": "TAG_FLAG",
    "HA": "TAG_HA",
    "3xHA": "TAG_HA",
    "Myc": "TAG_MYC",
    "c-Myc": "TAG_MYC",
    "V5 tag": "TAG_V5",
    "V5": "TAG_V5",
    "Strep-Tag II": "TAG_STREP",
    "Strep-tag II": "TAG_STREP",
    "GST": "TAG_GST",
    "MBP": "TAG_MBP",
    # --- Reporters ---
    "GFP": "REPORTER_GFP",
    "EGFP": "REPORTER_EGFP",
    "mEGFP": "REPORTER_EGFP",
    "eGFP": "REPORTER_EGFP",
    "mCherry": "REPORTER_MCHERRY",
    "mCherry2": "REPORTER_MCHERRY",
    "RFP": "REPORTER_RFP",
    "TagRFP": "REPORTER_RFP",
    "TagRFP-T": "REPORTER_RFP",
    "mRFP1": "REPORTER_RFP",
    "mRuby": "REPORTER_MRUBY",
    "mRuby2": "REPORTER_MRUBY",
    "BFP": "REPORTER_BFP",
    "EBFP2": "REPORTER_BFP",
    "mTagBFP2": "REPORTER_BFP",
    "YFP": "REPORTER_YFP",
    "EYFP": "REPORTER_YFP",
    "mCitrine": "REPORTER_YFP",
    "mVenus": "REPORTER_YFP",
    "CFP": "REPORTER_CFP",
    "ECFP": "REPORTER_CFP",
    "mCerulean3": "REPORTER_CFP",
    "luciferase": "REPORTER_LUCIFERASE",
    "Firefly luciferase": "REPORTER_LUCIFERASE",
    "Renilla luciferase": "REPORTER_LUCIFERASE",
    "lacZ": "REPORTER_LACZ",
    "lacZα": "REPORTER_LACZ",
    # --- Elements ---
    "IRES": "ELEM_IRES",
    "IRES2": "ELEM_IRES",
    "WPRE": "ELEM_WPRE",
    "cPPT/CTS": "ELEM_CPPT",
    "cPPT": "ELEM_CPPT",
    "5' LTR": "ELEM_LTR_5",
    "5' LTR (truncated)": "ELEM_LTR_5",
    "3' LTR (ΔU3)": "ELEM_LTR_3",
    "3' LTR": "ELEM_LTR_3",
    "Ψ": "ELEM_PSI",
    "psi": "ELEM_PSI",
    "PSI": "ELEM_PSI",
    "SV40 poly(A) signal": "ELEM_POLYA_SV40",
    "SV40 polyA": "ELEM_POLYA_SV40",
    "SV40 pA": "ELEM_POLYA_SV40",
    "bGH poly(A) signal": "ELEM_POLYA_BGH",
    "BGH pA": "ELEM_POLYA_BGH",
    "BGH poly(A) signal": "ELEM_POLYA_BGH",
    "CMV enhancer": "ELEM_CMV_ENHANCER",
    "gRNA scaffold": "ELEM_GRNA_SCAFFOLD",
    "sgRNA scaffold": "ELEM_GRNA_SCAFFOLD",
    "tracrRNA": "ELEM_TRACRRNA",
    "T7 terminator": "ELEM_TERM_T7",
    "rrnB T1 terminator": "ELEM_TERM_RRNB",
    "rrnB T2 terminator": "ELEM_TERM_RRNB",
    "CYC1 terminator": "ELEM_TERM_CYC1",
    "P2A": "ELEM_P2A",
    "T2A": "ELEM_T2A",
    "E2A": "ELEM_E2A",
    "F2A": "ELEM_F2A",
    "loxP": "ELEM_LOXP",
    "FRT": "ELEM_FRT",
    "attB": "ELEM_ATTB",
    "attP": "ELEM_ATTP",
}

# Build reverse lookup: token_inner → set of feature names (for logging)
_TOKEN_TO_FEATURES: dict[str, set[str]] = {}
for _feat, _tok_inner in _FEATURE_TO_TOKEN.items():
    _TOKEN_TO_FEATURES.setdefault(_tok_inner, set()).add(_feat)

# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------

print(f"Loading {MODEL_ID} on {DEVICE} …")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
if DEVICE == "cuda":
    model = model.half().to("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
model.eval()

vocab = tokenizer.get_vocab()

if tokenizer.eos_token_id is None and "<EOS>" in vocab:
    tokenizer.eos_token_id = vocab["<EOS>"]
if tokenizer.pad_token_id is None and "<PAD>" in vocab:
    tokenizer.pad_token_id = vocab["<PAD>"]

SPECIAL_TOKENS = sorted(
    tok for tok in vocab
    if tok.startswith("<") and tok.endswith(">")
    and any(tok.strip("<>").startswith(p) for p in FUNCTIONAL_PREFIXES)
)

TOKEN_BY_CATEGORY: dict[str, list[str]] = {}
for _tok in SPECIAL_TOKENS:
    _cat = _tok.strip("<>").split("_", 1)[0]
    TOKEN_BY_CATEGORY.setdefault(_cat, []).append(_tok)

import transformers as _tf
print(f"Ready: {len(SPECIAL_TOKENS)} tokens, device={DEVICE}, "
      f"transformers={_tf.__version__}, torch={torch.__version__}")

# ---------------------------------------------------------------------------
# pLannotate database setup (background)
# ---------------------------------------------------------------------------

_plannotate_ready = threading.Event()


def _init_plannotate_db():
    try:
        if not _plannotate_rsc.databases_exist():
            print("Downloading pLannotate databases…")
            _plannotate_rsc.download_databases()
        _plannotate_ready.set()
        print("pLannotate database ready.")
    except Exception as exc:
        print(f"pLannotate DB setup failed: {exc}")
        _plannotate_ready.set()


threading.Thread(target=_init_plannotate_db, daemon=True).start()

# ---------------------------------------------------------------------------
# Token-mapping helpers
# ---------------------------------------------------------------------------


def _build_token_list_for_prompt() -> str:
    lines = []
    for cat in sorted(TOKEN_BY_CATEGORY):
        lines.append(f"  {cat}: {', '.join(TOKEN_BY_CATEGORY[cat])}")
    return "\n".join(lines)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a molecular biology expert. Convert the user's plasmid description "
    "into structured tokens for the PlasmidLM DNA generation model.\n\n"
    "Available tokens by category:\n{token_list}\n\n"
    "Rules:\n"
    "1. Output ONLY space-separated tokens in angle brackets. No prose, no explanation.\n"
    "2. Pick the most specific token for each requested component.\n"
    "3. Skip anything that does not map to an available token.\n"
    "4. Always infer standard companions:\n"
    "   - Bacterial vectors need <ORI_COLE1> and an <AMR_*> if not specified\n"
    "   - Lentiviral vectors typically need <ELEM_CPPT> <ELEM_LTR_5> <ELEM_PSI> <ELEM_WPRE>\n"
    "   - Most vectors need <PROM_AMPR> (AmpR promoter for the resistance cassette)\n"
    "   - CRISPR vectors need <ELEM_GRNA_SCAFFOLD> <ELEM_TRACRRNA>\n"
    "   - Include <ELEM_IRES> and <ELEM_TRACRRNA> for most bacterial constructs\n"
    "5. Order: VEC_ first, then SP_, COPY_, BB_, AMR_, ORI_, PROM_, ELEM_, TAG_, REPORTER_\n\n"
    "Examples (from real training data):\n"
    '"bacterial cloning vector with kanamycin" → '
    "<VEC_BACTERIAL> <AMR_KANAMYCIN> <ORI_COLE1>\n"
    '"lentiviral GFP with puromycin selection" → '
    "<VEC_LENTIVIRAL> <VEC_MAMMALIAN> <AMR_AMPICILLIN> <AMR_PUROMYCIN> <ORI_COLE1> "
    "<PROM_AMPR> <PROM_CMV> <ELEM_CPPT> <ELEM_LTR_5> <ELEM_POLYA_SV40> <ELEM_PSI> "
    "<ELEM_WPRE> <REPORTER_GFP>\n"
    '"CRISPR guide RNA vector with U6 promoter and ampicillin" → '
    "<VEC_CRISPR> <SP_HUMAN> <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> <PROM_U6> "
    "<ELEM_GRNA_SCAFFOLD> <ELEM_IRES> <ELEM_TRACRRNA>\n"
    '"mammalian expression with CMV, EGFP, neomycin" → '
    "<VEC_MAMMALIAN> <SP_HUMAN> <AMR_KANAMYCIN> <ORI_COLE1> <ORI_SV40> <PROM_CMV> "
    "<PROM_SV40> <ELEM_CMV_ENHANCER> <ELEM_POLYA_BGH> <REPORTER_EGFP>\n"
    '"bacterial T7 expression with His-tag for E. coli" → '
    "<VEC_BACTERIAL> <SP_ECOLI> <COPY_HIGH> <AMR_AMPICILLIN> <ORI_COLE1> <PROM_AMPR> "
    "<PROM_LAC> <PROM_T7> <ELEM_IRES> <ELEM_TRACRRNA> <TAG_HIS>\n"
)


def _load_system_prompt() -> str:
    path = Path(__file__).parent / "prompt.txt"
    if path.exists():
        text = path.read_text().strip()
        if "{token_list}" in text:
            return text.format(token_list=_build_token_list_for_prompt())
        return text
    return _DEFAULT_SYSTEM_PROMPT.format(
        token_list=_build_token_list_for_prompt()
    )


def map_text_to_tokens(description: str) -> tuple[str, str]:
    """Map free-text description → validated token prompt via Claude Haiku."""
    if not description.strip():
        return "", "Enter a plasmid description first."

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "", "ANTHROPIC_API_KEY not set."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=256,
            system=_load_system_prompt(),
            messages=[{"role": "user", "content": description}],
        )
        raw = resp.content[0].text.strip()
    except Exception as exc:
        return "", f"LLM call failed: {exc}"

    found = re.findall(r"<[^>]+>", raw)
    valid, invalid = [], []
    for t in found:
        (valid if t in vocab else invalid).append(t)

    if not valid:
        return "", f"No valid tokens in LLM output: {raw}"

    _CAT_ORDER = {
        "VEC": 0, "SP": 1, "COPY": 2, "BB": 3, "AMR": 4,
        "ORI": 5, "PROM": 6, "ELEM": 7, "TAG": 8, "REPORTER": 9,
    }
    seen = set()
    unique = []
    for t in valid:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    def _sort_key(tok: str) -> tuple[int, str]:
        inner = tok.strip("<>")
        cat = inner.split("_", 1)[0]
        return (_CAT_ORDER.get(cat, 99), inner)

    unique.sort(key=_sort_key)

    prompt = f"<BOS> {' '.join(unique)} <SEQ>"
    status = "Tokens mapped successfully."
    if invalid:
        status += f"  Removed invalid: {', '.join(invalid)}"
    return prompt, status


# ---------------------------------------------------------------------------
# DNA generation
# ---------------------------------------------------------------------------


def _ensure_prompt_format(text: str) -> str:
    text = text.strip()
    if not text.startswith("<BOS>"):
        text = "<BOS> " + text
    if not text.endswith("<SEQ>"):
        text = text.rstrip() + " <SEQ>"
    return text


def _clean_dna(raw: str) -> str:
    seq = re.sub(r"<[^>]+>", "", raw.upper())
    return re.sub(r"[^ATGCN]", "", seq)


def _parse_hard_tokens(prompt: str) -> list[str]:
    """Extract hard annotation tokens from a prompt string."""
    all_tokens = re.findall(r"<[^>]+>", prompt)
    return [
        t for t in all_tokens
        if any(t.strip("<>").startswith(p) for p in HARD_PREFIXES)
    ]


def _map_annotations_to_tokens(
    hits: pd.DataFrame,
) -> dict[str, dict]:
    """Map plannotate annotations back to PlasmidLM token names.

    Returns {token_inner: {"percmatch": float, "feature": str}} for
    the best hit per token.
    """
    found: dict[str, dict] = {}
    for _, row in hits.iterrows():
        feature = str(row.get("Feature", ""))
        token_inner = _FEATURE_TO_TOKEN.get(feature)
        if token_inner is None:
            # Try case-insensitive / partial matching
            feat_lower = feature.lower().strip()
            for key, val in _FEATURE_TO_TOKEN.items():
                if key.lower() == feat_lower:
                    token_inner = val
                    break
            if token_inner is None:
                continue
        pm = float(row.get("percmatch", 0) or 0)
        prev = found.get(token_inner)
        if prev is None or pm > prev["percmatch"]:
            found[token_inner] = {"percmatch": pm, "feature": feature}
    return found


def _score_annotation(
    hits: pd.DataFrame | None,
    prompt: str = "",
    recall_floor: float = 0.5,
    dup_origin_penalty: float = 0.85,
    dup_element_penalty: float = 0.95,
) -> float:
    """Score annotation by mapping results back to requested prompt tokens.

    Uses the same composite formula as the GRPO plannotate scorer:
      quality = geo_mean(per-token scores)
      recall  = found / expected
      composite = quality * recall_penalty * duplication_penalties
    """
    if hits is None or (hasattr(hits, "empty") and hits.empty):
        return 0.0

    expected_tokens = _parse_hard_tokens(prompt)
    if not expected_tokens:
        return sum(
            float(row.get("percmatch", 0) or 0) / 100.0
            for _, row in hits.iterrows()
        )

    mapped = _map_annotations_to_tokens(hits)
    found_scores = []
    for tok in expected_tokens:
        tok_inner = tok.strip("<>")
        if tok_inner in mapped:
            found_scores.append(mapped[tok_inner]["percmatch"] / 100.0)

    n_expected = len(expected_tokens)
    n_found = len(found_scores)
    recall = n_found / n_expected if n_expected > 0 else 0.0

    if found_scores:
        log_scores = [math.log(max(s, 1e-6)) for s in found_scores]
        geo_mean = math.exp(sum(log_scores) / len(log_scores))
        quality = geo_mean ** 2  # sharpness = 2
    else:
        quality = 0.0

    recall_penalty = recall_floor + (1.0 - recall_floor) * recall
    composite = quality * recall_penalty

    # Penalize duplicate origins and duplicate elements
    # Count how many times each mapped token category appears in annotations
    token_counts: dict[str, int] = {}
    for _, row in hits.iterrows():
        feature = str(row.get("Feature", ""))
        tok_inner = _FEATURE_TO_TOKEN.get(feature)
        if tok_inner is None:
            feat_lower = feature.lower().strip()
            for key, val in _FEATURE_TO_TOKEN.items():
                if key.lower() == feat_lower:
                    tok_inner = val
                    break
        if tok_inner:
            token_counts[tok_inner] = token_counts.get(tok_inner, 0) + 1

    # Count excess origins (more than expected)
    expected_origins = sum(1 for t in expected_tokens if t.strip("<>").startswith("ORI_"))
    found_origins = sum(v for k, v in token_counts.items() if k.startswith("ORI_"))
    excess_origins = max(0, found_origins - max(expected_origins, 1))
    if excess_origins > 0:
        composite *= dup_origin_penalty ** excess_origins

    # Count duplicate elements (same element appearing multiple times)
    for tok_inner, count in token_counts.items():
        if tok_inner.startswith("ELEM_") and count > 1:
            composite *= dup_element_penalty ** (count - 1)

    print(f"[score] recall={n_found}/{n_expected}={recall:.2f}, "
          f"quality={quality:.3f}, composite={composite:.3f}")
    return composite


class _TokenCounter:
    """Counts generation steps so the UI can show progress.

    Implements the streamer interface (put/end) expected by
    transformers model.generate().
    """

    def __init__(self):
        self.step = 0
        self.done = False

    def put(self, value):
        self.step += 1

    def end(self):
        self.done = True


def _progress_bar(step: int, total: int, width: int = 20) -> str:
    frac = min(step / max(total, 1), 1.0)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {step}/{total} tokens ({frac:.0%})"


def generate_and_select(
    prompt_text: str,
    temperature: float,
    num_samples: int,
    max_tokens: int,
):
    """Generate N samples in a batch, annotate each, return the best."""
    if not prompt_text.strip():
        yield "", "Please provide a token prompt first.", None, None, ""
        return

    prompt_text = _ensure_prompt_format(prompt_text)
    num_samples = max(1, int(num_samples))
    max_tokens = int(max_tokens)
    print(f"[generate] prompt: {prompt_text!r}, n={num_samples}, "
          f"temp={temperature}, max_tokens={max_tokens}")

    # Batch generate: replicate input N times
    inputs = tokenizer(
        [prompt_text] * num_samples,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    # Run generation in background thread with token counter
    counter = _TokenCounter()
    result_holder: list = [None, None]  # [outputs, error]

    def _run_generate():
        try:
            with torch.no_grad():
                result_holder[0] = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=float(temperature),
                    do_sample=True,
                    top_k=50,
                    use_cache=True,
                    streamer=counter,
                )
        except Exception as exc:
            result_holder[1] = exc

    t0 = time.time()
    gen_thread = threading.Thread(target=_run_generate)
    gen_thread.start()

    # Poll progress and yield status updates
    n_label = f"{num_samples} sample(s)" if num_samples > 1 else "1 sample"
    while gen_thread.is_alive():
        elapsed = time.time() - t0
        bar = _progress_bar(counter.step, max_tokens)
        yield "", f"Generating {n_label}… {bar}  ({elapsed:.1f}s)", None, None, ""
        gen_thread.join(timeout=0.4)

    gen_time = time.time() - t0

    if result_holder[1] is not None:
        print(f"[generate] ERROR: {result_holder[1]}")
        yield "", f"Generation failed: {result_holder[1]}", None, None, ""
        return

    outputs = result_holder[0]

    # Decode all samples
    samples = []
    for i in range(outputs.shape[0]):
        raw = tokenizer.decode(outputs[i].tolist())
        dna_part = raw.split("<SEQ>")[-1] if "<SEQ>" in raw else raw
        dna = _clean_dna(dna_part)
        has_eos = "<EOS>" in raw
        samples.append((dna, has_eos))
        print(f"[generate] sample {i}: {len(dna)} bp, "
              f"{'complete' if has_eos else 'truncated'}")

    yield "", (f"Generated {n_label} ({counter.step} tokens) in {gen_time:.1f}s. "
               "Annotating…"), None, None, ""

    # If only 1 sample, skip scoring
    if num_samples == 1:
        dna, has_eos = samples[0]
        tag = "complete" if has_eos else "max-length"
        html_map, table, ann_status = _annotate(dna)
        status = f"{len(dna)} bp ({tag}, {gen_time:.1f}s). {ann_status}"
        yield dna, status, html_map, table, ""
        return

    # Annotate all samples in parallel
    _plannotate_ready.wait(timeout=30)
    yield "", f"Annotating {num_samples} samples in parallel…", None, None, ""

    def _annotate_and_score(idx: int) -> tuple[int, float]:
        dna_i = samples[idx][0]
        if len(dna_i) < 100:
            return idx, 0.0
        try:
            hits = _plannotate_annotate(dna_i, is_detailed=True, linear=False)
            score = _score_annotation(hits, prompt=prompt_text)
        except Exception:
            score = 0.0
        print(f"[score] sample {idx}: score={score:.2f}")
        return idx, score

    best_idx, best_score = 0, -1.0
    with ThreadPoolExecutor(max_workers=num_samples) as pool:
        futures = [pool.submit(_annotate_and_score, i) for i in range(num_samples)]
        for fut in as_completed(futures):
            idx, score = fut.result()
            if score > best_score:
                best_score = score
                best_idx = idx

    dna, has_eos = samples[best_idx]
    tag = "complete" if has_eos else "max-length"
    html_map, table, ann_status = _annotate(dna)
    status = (f"Best of {num_samples}: sample {best_idx+1}, "
              f"score={best_score:.2f}, "
              f"{len(dna)} bp ({tag}, {gen_time:.1f}s). {ann_status}")
    yield dna, status, html_map, table, ""


def _annotate(
    dna: str,
    min_percmatch: float = 90.0,
) -> tuple[str | None, pd.DataFrame | None, str]:
    """Run pLannotate on a DNA sequence, filtering to high-confidence hits."""
    if not dna or len(dna) < 100:
        return None, None, "Sequence too short for annotation."

    if not _plannotate_ready.wait(timeout=5):
        return None, None, "pLannotate DB still loading."

    try:
        hits = _plannotate_annotate(dna, is_detailed=True, linear=False)
    except Exception as exc:
        print(f"[annotate] ERROR: {exc}")
        return None, None, f"Annotation error: {exc}"

    if hits is None or (hasattr(hits, "empty") and hits.empty):
        return None, None, "No annotations found."

    # Filter to high-confidence annotations
    n_total = len(hits)
    if "percmatch" in hits.columns:
        hits = hits[hits["percmatch"] >= min_percmatch].copy()
    if hits.empty:
        return None, None, f"No annotations above {min_percmatch:.0f}% match ({n_total} below threshold)."

    html_map = None
    try:
        fig = _plannotate_bokeh(hits, linear=False)
        raw_html = file_html(fig, BOKEH_CDN, "Plasmid Map")
        import html as _html_mod
        escaped = _html_mod.escape(raw_html)
        html_map = (
            f'<iframe srcdoc="{escaped}" '
            f'style="width:100%;height:600px;border:none;" '
            f'sandbox="allow-scripts allow-same-origin"></iframe>'
        )
    except Exception as exc:
        print(f"[annotate] bokeh error: {exc}")

    display_cols = [
        c for c in ("Feature", "Type", "Description",
                     "percmatch", "length", "Start", "End", "Strand")
        if c in hits.columns
    ]
    table = hits[display_cols].copy() if display_cols else hits.copy()
    return html_map, table, f"Found {len(hits)} annotation(s) (>={min_percmatch:.0f}% match, {n_total - len(hits)} filtered)."


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _token_reference_md() -> str:
    lines = ["| Category | Tokens |", "| --- | --- |"]
    for cat in sorted(TOKEN_BY_CATEGORY):
        tokens = TOKEN_BY_CATEGORY[cat]
        lines.append(
            f"| **{cat}** | {', '.join(f'`{t}`' for t in tokens)} |"
        )
    return "\n".join(lines)


QUICK_START_EXAMPLES: list[tuple[str, str]] = [
    (
        "Simple bacterial cloning vector — kanamycin resistance, ColE1 origin",
        "<BOS> <VEC_BACTERIAL> <AMR_KANAMYCIN> <ORI_COLE1> <SEQ>",
    ),
    (
        "Lentiviral GFP reporter with CMV promoter and puromycin selection",
        "<BOS> <VEC_LENTIVIRAL> <VEC_MAMMALIAN> <AMR_AMPICILLIN> <AMR_PUROMYCIN> "
        "<ORI_COLE1> <PROM_AMPR> <PROM_CMV> <ELEM_CPPT> <ELEM_LTR_5> "
        "<ELEM_POLYA_SV40> <ELEM_PSI> <ELEM_WPRE> <REPORTER_GFP> <SEQ>",
    ),
    (
        "Human CRISPR guide RNA vector with U6 promoter and ampicillin resistance",
        "<BOS> <VEC_CRISPR> <SP_HUMAN> <AMR_AMPICILLIN> <ORI_COLE1> "
        "<PROM_AMPR> <PROM_U6> <ELEM_GRNA_SCAFFOLD> <ELEM_IRES> "
        "<ELEM_TRACRRNA> <SEQ>",
    ),
    (
        "Mammalian EGFP expression vector — CMV/SV40 dual promoter, neomycin selection",
        "<BOS> <VEC_MAMMALIAN> <SP_HUMAN> <AMR_KANAMYCIN> <ORI_COLE1> "
        "<ORI_SV40> <PROM_CMV> <PROM_SV40> <ELEM_CMV_ENHANCER> "
        "<ELEM_POLYA_BGH> <REPORTER_EGFP> <SEQ>",
    ),
    (
        "E. coli T7 protein expression with His-tag, lac-inducible, high copy",
        "<BOS> <VEC_BACTERIAL> <SP_ECOLI> <COPY_HIGH> <AMR_AMPICILLIN> "
        "<ORI_COLE1> <PROM_AMPR> <PROM_LAC> <PROM_T7> <ELEM_IRES> "
        "<ELEM_TRACRRNA> <TAG_HIS> <SEQ>",
    ),
]

_EXAMPLE_LABELS = [label for label, _ in QUICK_START_EXAMPLES]


def _use_quick_example(choice: str) -> tuple[str, str]:
    if not choice:
        return "", ""
    for label, tokens in QUICK_START_EXAMPLES:
        if label == choice:
            return tokens, f"Loaded: {label}"
    return "", "Example not found."

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
.preset-dropdown input {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    height: auto !important;
    min-height: 2.4em;
}
.preset-dropdown .options .item {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.4;
    padding: 8px 12px !important;
}
.ann-table td {
    white-space: normal !important;
    word-wrap: break-word !important;
    max-width: 300px;
}
"""

with gr.Blocks(title="PlasmidSpace", css=_CUSTOM_CSS) as demo:
    gr.Markdown(
        "# \U0001F9EC PlasmidSpace\n"
        "*Design synthetic plasmids from natural language using "
        "[PlasmidLM]"
        "(https://huggingface.co/McClain/PlasmidLM-kmer6-GRPO-plannotate)*"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            with gr.Tabs():
                with gr.Tab("Quick Start"):
                    quick_start = gr.Dropdown(
                        choices=_EXAMPLE_LABELS,
                        label="Choose a preset plasmid",
                        value=None,
                        interactive=True,
                        elem_classes=["preset-dropdown"],
                    )
                with gr.Tab("Describe Your Own"):
                    description = gr.Textbox(
                        label="Describe your plasmid",
                        placeholder=(
                            "e.g. kanamycin resistance with ColE1 origin "
                            "and T7 promoter for bacterial expression"
                        ),
                        lines=2,
                    )
                    map_btn = gr.Button("Map to Tokens", variant="secondary")

            token_prompt = gr.Textbox(
                label="Token Prompt (editable)",
                placeholder="<BOS> <AMR_KANAMYCIN> <ORI_COLE1> <PROM_T7> <SEQ>",
                lines=2,
                info="Edit freely, or type tokens manually.",
            )

            with gr.Row():
                temperature = gr.Slider(
                    0.1, 1.0, value=0.3, step=0.05, label="Temperature",
                )
                num_samples = gr.Slider(
                    1, 8, value=3, step=1, label="Top N",
                )

            max_tokens = gr.Slider(
                500, 5000, value=3000, step=100, label="Max Tokens",
            )

            generate_btn = gr.Button(
                "Generate Plasmid", variant="primary", size="lg",
            )

            with gr.Accordion("Available Tokens", open=False):
                gr.Markdown(_token_reference_md())

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Status", interactive=False, max_lines=2,
            )
            with gr.Tabs():
                with gr.Tab("Plasmid Map"):
                    plasmid_html = gr.HTML(label="Plasmid Map")
                with gr.Tab("Annotations"):
                    ann_table = gr.Dataframe(
                        label="Annotation Table",
                        wrap=True,
                        elem_classes=["ann-table"],
                    )
                with gr.Tab("DNA Sequence"):
                    dna_output = gr.Textbox(
                        label="Generated DNA",
                        lines=12, max_lines=25, interactive=False,
                    )

    # Hidden dummy for 5th output of generator
    _dummy = gr.Textbox(visible=False)

    # ── Event wiring ──────────────────────────────────────────────

    quick_start.change(
        fn=_use_quick_example,
        inputs=[quick_start],
        outputs=[token_prompt, status_box],
    ).then(
        fn=generate_and_select,
        inputs=[token_prompt, temperature, num_samples, max_tokens],
        outputs=[dna_output, status_box, plasmid_html, ann_table, _dummy],
    )

    map_btn.click(
        fn=map_text_to_tokens,
        inputs=[description],
        outputs=[token_prompt, status_box],
    )

    generate_btn.click(
        fn=generate_and_select,
        inputs=[token_prompt, temperature, num_samples, max_tokens],
        outputs=[dna_output, status_box, plasmid_html, ann_table, _dummy],
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
