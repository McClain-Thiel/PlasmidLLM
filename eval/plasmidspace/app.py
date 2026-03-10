"""PlasmidSpace – AI-powered plasmid design demo.

Generates synthetic plasmid DNA from natural language descriptions using
PlasmidLM, then annotates with pLannotate for visualization.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Streamlit shim — plannotate imports streamlit at the top level but we only
# use the annotation/plot modules, not the streamlit UI.
# ---------------------------------------------------------------------------
import importlib
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
    # plannotate.pLannotate imports streamlit.cli
    _st_cli = types.ModuleType("streamlit.cli")
    sys.modules["streamlit.cli"] = _st_cli

# ---------------------------------------------------------------------------

import os
import re
import time
from pathlib import Path

import anthropic
import gradio as gr
import pandas as pd
import torch
from bokeh.embed import file_html
from bokeh.resources import CDN as BOKEH_CDN
from plannotate.annotate import annotate as _plannotate_annotate
from plannotate.bokeh_plot import get_bokeh as _plannotate_bokeh
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn=None, **kwargs):
        return fn if fn is not None else lambda f: f

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "McClain/PlasmidLM-kmer6-GRPO-plannotate"

FUNCTIONAL_PREFIXES = frozenset(
    {"AMR_", "ORI_", "PROM_", "REPORTER_", "TAG_", "ELEM_",
     "BB_", "VEC_", "SP_", "COPY_"}
)

# ---------------------------------------------------------------------------
# Model & tokenizer (loaded once at startup)
# ---------------------------------------------------------------------------

print(f"Loading {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
)
model.eval()

vocab = tokenizer.get_vocab()

SPECIAL_TOKENS = sorted(
    tok for tok in vocab
    if tok.startswith("<") and tok.endswith(">")
    and any(tok.strip("<>").startswith(p) for p in FUNCTIONAL_PREFIXES)
)

TOKEN_BY_CATEGORY: dict[str, list[str]] = {}
for _tok in SPECIAL_TOKENS:
    _cat = _tok.strip("<>").split("_", 1)[0]
    TOKEN_BY_CATEGORY.setdefault(_cat, []).append(_tok)

print(
    f"Ready: {len(SPECIAL_TOKENS)} functional tokens, "
    f"{len(TOKEN_BY_CATEGORY)} categories"
)

# ---------------------------------------------------------------------------
# pLannotate database setup (background, non-blocking)
# ---------------------------------------------------------------------------

import threading
from plannotate import resources as _plannotate_rsc

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

    # De-duplicate and sort by category order (matching training data convention)
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
# DNA generation (streaming, GPU-accelerated)
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


@gpu
def generate_dna(
    prompt_text: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
):
    """Generate DNA. Returns ``(dna, status)`` tuple."""
    if not prompt_text.strip():
        return "", "Please provide a token prompt first."

    prompt_text = _ensure_prompt_format(prompt_text)
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(dev)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=True,
            top_k=int(top_k),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    raw_output = tokenizer.decode(output[0], skip_special_tokens=False)
    dna_part = raw_output.split("<SEQ>")[-1] if "<SEQ>" in raw_output else raw_output
    dna = _clean_dna(dna_part)
    has_eos = "<EOS>" in raw_output
    tag = "complete" if has_eos else "max-length reached"
    return dna, f"Done: {len(dna)} bp, {tag} ({elapsed:.1f} s)"


# ---------------------------------------------------------------------------
# Annotation & visualisation
# ---------------------------------------------------------------------------


def annotate_sequence(
    dna: str,
) -> tuple[str | None, pd.DataFrame | None, str]:
    """Run pLannotate on *dna*, returning ``(html_map, table_df, status)``."""
    if not dna or len(dna) < 100:
        return None, None, "Sequence too short for annotation (need >= 100 bp)."

    if not _plannotate_ready.wait(timeout=5):
        return None, None, "pLannotate database still loading — try again shortly."

    try:
        hits = _plannotate_annotate(dna, is_detailed=True, linear=False)
    except Exception as exc:
        return None, None, f"Annotation error: {exc}"

    if hits is None or (hasattr(hits, "empty") and hits.empty):
        return None, None, "No annotations found."

    # Bokeh plasmid map → standalone HTML
    html_map = None
    try:
        fig = _plannotate_bokeh(hits, linear=False)
        html_map = file_html(fig, BOKEH_CDN, "Plasmid Map")
    except Exception:
        pass

    display_cols = [
        c
        for c in (
            "Feature", "Type", "Description",
            "percmatch", "length", "Start", "End", "Strand",
        )
        if c in hits.columns
    ]
    table = hits[display_cols].copy() if display_cols else hits.copy()

    return html_map, table, f"Found {len(hits)} annotation(s)."


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


EXAMPLE_DESCRIPTIONS = [
    ["Bacterial expression vector with kanamycin resistance, "
     "ColE1 origin, and T7 promoter"],
    ["Lentiviral vector with GFP reporter and puromycin selection"],
    ["CRISPR guide RNA vector with ampicillin resistance "
     "and U6 promoter"],
    ["Mammalian expression plasmid with CMV promoter, EGFP, "
     "and neomycin selection"],
]

# Pre-mapped examples from training data — bypass Claude, fill tokens directly.
QUICK_START_EXAMPLES: list[tuple[str, str]] = [
    (
        "Bacterial cloning vector (kanamycin, ColE1)",
        "<BOS> <VEC_BACTERIAL> <AMR_KANAMYCIN> <ORI_COLE1> <SEQ>",
    ),
    (
        "Lentiviral GFP with puromycin selection",
        "<BOS> <VEC_LENTIVIRAL> <VEC_MAMMALIAN> <AMR_AMPICILLIN> <AMR_PUROMYCIN> "
        "<ORI_COLE1> <PROM_AMPR> <PROM_CMV> <ELEM_CPPT> <ELEM_LTR_5> "
        "<ELEM_POLYA_SV40> <ELEM_PSI> <ELEM_WPRE> <REPORTER_GFP> <SEQ>",
    ),
    (
        "CRISPR guide RNA vector (U6, ampicillin)",
        "<BOS> <VEC_CRISPR> <SP_HUMAN> <AMR_AMPICILLIN> <ORI_COLE1> "
        "<PROM_AMPR> <PROM_U6> <ELEM_GRNA_SCAFFOLD> <ELEM_IRES> "
        "<ELEM_TRACRRNA> <SEQ>",
    ),
    (
        "Mammalian expression (CMV, EGFP, neomycin)",
        "<BOS> <VEC_MAMMALIAN> <SP_HUMAN> <AMR_KANAMYCIN> <ORI_COLE1> "
        "<ORI_SV40> <PROM_CMV> <PROM_SV40> <ELEM_CMV_ENHANCER> "
        "<ELEM_POLYA_BGH> <REPORTER_EGFP> <SEQ>",
    ),
    (
        "Bacterial T7 expression with His-tag",
        "<BOS> <VEC_BACTERIAL> <SP_ECOLI> <COPY_HIGH> <AMR_AMPICILLIN> "
        "<ORI_COLE1> <PROM_AMPR> <PROM_LAC> <PROM_T7> <ELEM_IRES> "
        "<ELEM_TRACRRNA> <TAG_HIS> <SEQ>",
    ),
]


def _use_quick_example(choice: str) -> tuple[str, str]:
    """Look up pre-mapped tokens for a quick-start example."""
    for label, tokens in QUICK_START_EXAMPLES:
        if label == choice:
            return tokens, f"Loaded preset: {label}"
    return "", "Example not found."

# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="PlasmidSpace") as demo:
    gr.Markdown(
        "# \U0001F9EC PlasmidSpace\n"
        "*Design synthetic plasmids from natural language using "
        "[PlasmidLM]"
        "(https://huggingface.co/McClain/PlasmidLM-kmer6-GRPO-plannotate)*"
    )

    with gr.Row(equal_height=False):
        # ── Left column: inputs ───────────────────────────────────
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("### Quick Start")
            quick_start = gr.Radio(
                choices=[label for label, _ in QUICK_START_EXAMPLES],
                label="Pick a preset (skips token mapping)",
                value=None,
            )

            gr.Markdown("### Or describe your own")
            description = gr.Textbox(
                label="Describe your plasmid",
                placeholder=(
                    "e.g. kanamycin resistance with ColE1 origin "
                    "and T7 promoter for bacterial expression"
                ),
                lines=3,
            )
            map_btn = gr.Button("Map to Tokens", variant="secondary")

            with gr.Row():
                temperature = gr.Slider(
                    0.1, 1.0, value=0.3, step=0.05,
                    label="Temperature",
                )
                top_k = gr.Slider(
                    1, 100, value=50, step=1, label="Top K",
                )

            max_tokens = gr.Slider(
                500, 5000, value=3000, step=100, label="Max Tokens",
            )

            token_prompt = gr.Textbox(
                label="Token Prompt (editable)",
                placeholder=(
                    "<BOS> <AMR_KANAMYCIN> <ORI_COLE1> <PROM_T7> <SEQ>"
                ),
                lines=2,
                info="Edit freely, or type tokens manually and skip mapping.",
            )

            generate_btn = gr.Button(
                "Generate Plasmid", variant="primary", size="lg",
            )

            with gr.Accordion("Available Tokens", open=False):
                gr.Markdown(_token_reference_md())

        # ── Right column: outputs ─────────────────────────────────
        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Status", interactive=False, max_lines=2,
            )

            dna_output = gr.Textbox(
                label="Generated DNA Sequence",
                lines=10,
                max_lines=20,
                interactive=False,
            )

            with gr.Tabs():
                with gr.Tab("Plasmid Map"):
                    plasmid_html = gr.HTML(label="Plasmid Map")
                with gr.Tab("Annotations"):
                    ann_table = gr.Dataframe(label="Annotation Table")

    # ── Event wiring ──────────────────────────────────────────────

    # Quick start: fill tokens → generate → annotate
    quick_start.change(
        fn=_use_quick_example,
        inputs=[quick_start],
        outputs=[token_prompt, status_box],
    ).then(
        fn=generate_dna,
        inputs=[token_prompt, temperature, top_k, max_tokens],
        outputs=[dna_output, status_box],
    ).then(
        fn=annotate_sequence,
        inputs=[dna_output],
        outputs=[plasmid_html, ann_table, status_box],
    )

    # Manual: describe → map → edit → generate → annotate
    map_btn.click(
        fn=map_text_to_tokens,
        inputs=[description],
        outputs=[token_prompt, status_box],
    )

    generate_btn.click(
        fn=generate_dna,
        inputs=[token_prompt, temperature, top_k, max_tokens],
        outputs=[dna_output, status_box],
    ).then(
        fn=annotate_sequence,
        inputs=[dna_output],
        outputs=[plasmid_html, ann_table, status_box],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
