# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23",
#     "pandas==2.3.3",
#     "pyarrow>=14.0",
#     "numpy==2.2.6",
#     "statsmodels==0.14.6",
#     "matplotlib==3.10.8",
#     "huggingface_hub>=1.0",
#     "tabulate>=0.9",
# ]
# ///
"""Soft-token conditioning analysis — base vs GRPO on matched paired runs.

The question
------------
The prompt vocabulary splits into hard tokens (point motifs pLannotate can
BLAST-verify: AMR_, ORI_, PROM_, REPORTER_, REP_, TAG_, ELEM_) and soft tokens
(plasmid-level properties: BB_, COPY_, SP_, VEC_). The GRPO reward is defined
only over hard tokens — `eval/scripts/compute_extra_metrics.py:33-34`
explicitly names soft tokens "metadata, not pLannotate-verifiable, and
ignored." We want to know whether the pretrained base model conditions on
soft tokens at all. Whether GRPO shifted that conditioning is a secondary
sanity check — the reward never saw these tags, so the null is no change.

Method
------
For each soft value (e.g. `SP_HUMAN`) we list the hard tokens that biology
says should appear if the model conditioned on the tag (e.g. PROM_CMV,
ELEM_POLYA_BGH, ...). Each hard token resolves to a set of pLannotate sseqids
via the existing motif registry (`data/motif_registry_combined.parquet`).
A sequence "hits" a probe if its detected sseqid set intersects any of the
probe's sseqids. Citations for every probe rule are rendered inline.

Confound
--------
Hard tokens co-vary with soft tokens in the training distribution — an
SP_HUMAN prompt almost always also carries PROM_CMV as a hard token, which
independently pushes generation mammalian. We report marginal associations
only; we do not claim causal attribution to soft tokens.

Data provenance
---------------
Both runs live under `hf://buckets/McClain/PlasmidLMEval/runs/` and are
identical to those consumed by `notebooks/base_vs_grpo_analysis.py`:
    grpo_plannotate_full_20260408/     (generated 2026-04-08, annotated 2026-04-09)
    base_kmer6_matched_20260420_1303/  (generated + annotated 2026-04-20)

Prompts and sampling config are matched. All analysis reads from the local
`/tmp/plasmidlm_eval_cache/` populated by `hf buckets sync`.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.contingency_tables import mcnemar

    return mcnemar, mo, np, pd, plt, proportion_confint


@app.cell
def _(mo):
    mo.md(r"""
    # Soft-token conditioning — does the model use BB_ / COPY_ / SP_ / VEC_?

    **Paired design, same 1000 prompts.** Compares the pretrained base
    (`McClain/PlasmidLM-kmer6`) against the post-trained
    `McClain/PlasmidLM-kmer6-GRPO-plannotate`. The GRPO reward ignored soft
    tokens — so the headline question is whether the **base** model
    conditions on them at all. Base-vs-GRPO deltas are a sanity check:
    expected null.

    Each soft value is probed through **existing hard-token sseqid sets** in
    the motif registry — we just compose soft → hard → sseqid → detection.
    No new annotation, no external tools.

    Structure:

    1. Load matched runs + motif registry.
    2. Soft-token extraction from prompts.
    3. `SOFT_TO_HARD` probe map with citations.
    4. Null-buster: does base distinguish SP_HUMAN from SP_ECOLI at all?
    5. COPY_ block (4 buckets: correct / wrong / unclassified / no_signal).
    6. VEC_ block (heatmap, diagonal = correct conditioning).
    7. SP_ block (host-group heatmap; largely redundant with VEC_).
    8. BB_ block (per-backbone completeness distribution).
    9. Summary table.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import subprocess

    BUCKET = "McClain/PlasmidLMEval"
    GRPO_RUN = "runs/grpo_plannotate_full_20260408"
    BASE_RUN = "runs/base_kmer6_matched_20260420_1303"

    CACHE = Path("/tmp/plasmidlm_eval_cache")
    CACHE.mkdir(exist_ok=True)

    WANTED = [
        "generations.parquet",
        "plannotate_results.json",
        "metrics/tier3_essentials.parquet",
        "metrics/prompt_fidelity_per_seq.parquet",
    ]

    # The notebook reads from a local cache of bucket files. Populate with
    #   hf buckets sync hf://buckets/McClain/PlasmidLMEval/runs/grpo_plannotate_full_20260408 \
    #       /tmp/plasmidlm_eval_cache/runs__grpo_plannotate_full_20260408
    #   hf buckets sync hf://buckets/McClain/PlasmidLMEval/runs/base_kmer6_matched_20260420_1303 \
    #       /tmp/plasmidlm_eval_cache/runs__base_kmer6_matched_20260420_1303
    def _ensure_local(run_key, wanted):
        local_run = CACHE / run_key.replace("/", "__")
        local_run.mkdir(exist_ok=True, parents=True)
        missing = [f for f in wanted if not (local_run / f).exists()]
        if missing:
            src = f"hf://buckets/{BUCKET}/{run_key}"
            proc = subprocess.run(
                ["hf", "buckets", "sync", src, str(local_run)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Missing local files under {local_run} and `hf buckets sync` failed:\n"
                    f"{proc.stderr}\n"
                    "Populate the cache manually:\n"
                    f"    hf buckets sync {src} {local_run}"
                )
        return local_run

    grpo_dir = _ensure_local(GRPO_RUN, WANTED)
    base_dir = _ensure_local(BASE_RUN, WANTED)

    REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "motif_registry_combined.parquet"
    assert REGISTRY_PATH.exists(), f"motif registry not found at {REGISTRY_PATH}"
    return REGISTRY_PATH, base_dir, grpo_dir


@app.cell
def _(base_dir, grpo_dir, pd):
    import json

    def _load(d):
        gens = pd.read_parquet(d / "generations.parquet").reset_index(drop=True)
        tier3 = pd.read_parquet(d / "metrics/tier3_essentials.parquet").reset_index(drop=True)
        fidelity = pd.read_parquet(d / "metrics/prompt_fidelity_per_seq.parquet").reset_index(drop=True)
        with open(d / "plannotate_results.json") as f:
            pn = json.load(f)
        per_seq_sseqids = {}
        for entry in pn:
            sid = entry["seq_id"]
            per_seq_sseqids[sid] = {
                feat["sseqid"]
                for feat in entry.get("features", [])
                if isinstance(feat.get("sseqid"), str) and feat["sseqid"]
            }
        return {"gens": gens, "tier3": tier3, "fidelity": fidelity, "sseqids": per_seq_sseqids}

    base = _load(base_dir)
    grpo = _load(grpo_dir)

    assert len(base["gens"]) == len(grpo["gens"]) == 1000
    assert (base["gens"]["prompt"].values == grpo["gens"]["prompt"].values).all(), \
        "prompt order mismatch — runs are not paired"
    # seq_id convention is gen_<row> — align plannotate JSON to the generations row index.
    for _r in (base, grpo):
        _r["sseqids_by_row"] = [
            _r["sseqids"].get(f"gen_{i}", set()) for i in range(len(_r["gens"]))
        ]
    return base, grpo


@app.cell
def _(mo):
    mo.md("""
    ## Soft-token extraction

    Parse `<BB_*> <COPY_*> <SP_*> <VEC_*>` out of each prompt with a single regex.
    At most one of each per prompt in the training distribution; coverage per
    category is what matters for the downstream blocks.
    """)
    return


@app.cell
def _(base, pd):
    import re

    SOFT_RE = re.compile(r"<(BB|COPY|SP|VEC)_([^>]+)>")

    def _extract(prompts):
        bb, cp, sp, vec = [], [], [], []
        for p in prompts:
            found = {"BB": None, "COPY": None, "SP": None, "VEC": None}
            for cat, val in SOFT_RE.findall(p):
                if found[cat] is None:
                    found[cat] = val
            bb.append(found["BB"]); cp.append(found["COPY"])
            sp.append(found["SP"]); vec.append(found["VEC"])
        return bb, cp, sp, vec

    # Same prompt order in both runs (paired-invariance asserted above) — extract once.
    _bb, _cp, _sp, _vec = _extract(base["gens"]["prompt"].tolist())
    tags = pd.DataFrame({"bb": _bb, "copy": _cp, "sp": _sp, "vec": _vec})
    tags_summary = pd.DataFrame({
        "category": ["BB", "COPY", "SP", "VEC"],
        "present": [tags["bb"].notna().sum(), tags["copy"].notna().sum(),
                    tags["sp"].notna().sum(), tags["vec"].notna().sum()],
        "distinct_values": [tags["bb"].nunique(), tags["copy"].nunique(),
                            tags["sp"].nunique(), tags["vec"].nunique()],
    })
    tags_summary
    return (tags,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Probe map (SOFT → HARD → sseqid)

    Each soft value maps to the hard tokens that biology says must appear on a
    "correct" generation. Citations are kept inline so a reviewer can click
    through. The `UNCLASSIFIED_FOR` bucket names hard tokens that are *not*
    diagnostic on the target axis (e.g. `ORI_SV40` is a mammalian viral origin
    that requires large T-antigen — it is not on the bacterial HIGH/LOW
    copy-number axis), so sequences whose only family hit falls there are
    reported as `unclassified`, not as `wrong`.
    """)
    return


@app.cell
def _(pd):
    SOFT_TO_HARD = {
        "COPY": {
            "HIGH": {"ORI_COLE1", "ORI_RSF"},
            "LOW":  {"ORI_P15A", "ORI_PSC101"},
        },
        "VEC": {
            "MAMMALIAN":  {"PROM_CMV", "PROM_EF1A", "PROM_CAG",
                           "ELEM_POLYA_BGH", "ELEM_POLYA_SV40",
                           "ELEM_CMV_ENHANCER"},
            "BACTERIAL":  {"PROM_T7", "PROM_T5", "PROM_LAC"},
            "LENTIVIRAL": {"ELEM_LTR_3", "ELEM_LTR_5", "ELEM_PSI",
                           "ELEM_CPPT", "ELEM_WPRE"},
            "CRISPR":     {"ELEM_GRNA_SCAFFOLD", "ELEM_TRACRRNA", "PROM_U6"},
            "AAV":        {"ELEM_AAV_ITR"},
        },
        "SP": {
            "HUMAN": {"PROM_CMV", "PROM_EF1A", "ELEM_POLYA_BGH"},
            "MOUSE": {"PROM_CMV", "PROM_EF1A", "ELEM_POLYA_BGH"},
            "RAT":   {"PROM_CMV", "PROM_EF1A", "ELEM_POLYA_BGH"},
            "ECOLI": {"PROM_T7", "PROM_T5", "PROM_LAC"},
            "YEAST": {"ORI_2MU"},
        },
        "BB": {
            "PUC19":            {"AMR_AMPICILLIN", "ORI_COLE1"},
            "PCDNA3":           {"PROM_CMV", "ELEM_POLYA_BGH",
                                 "AMR_NEOMYCIN", "ORI_SV40"},
            "PET28A":           {"PROM_T7", "AMR_KANAMYCIN", "TAG_HIS"},
            "PEGFP_C1":         {"REPORTER_EGFP", "PROM_CMV", "AMR_KANAMYCIN"},
            "PEGFP_N1":         {"REPORTER_EGFP", "PROM_CMV", "AMR_KANAMYCIN"},
            "PLENTIGUIDE_PURO": {"ELEM_LTR_3", "ELEM_LTR_5",
                                 "AMR_PUROMYCIN", "PROM_U6",
                                 "ELEM_GRNA_SCAFFOLD"},
            "PDONR221":         {"AMR_CHLORAMPHENICOL"},
            "PDONR223":         {"AMR_CHLORAMPHENICOL"},
            "PAAV":             {"ELEM_AAV_ITR"},
            "PAAV2":            {"ELEM_AAV_ITR"},
        },
    }

    UNCLASSIFIED_FOR = {
        # Not on the bacterial HIGH/LOW axis: F1 is a single-stranded phage
        # packaging origin; SV40 is a mammalian viral ori requiring T antigen;
        # 2μ is a yeast episomal origin.
        "COPY": {"ORI_F1", "ORI_SV40", "ORI_2MU"},
    }

    # Which broad hard-token prefixes a category's "family" covers — used by the
    # bucket classifier to say "the model generated a family-relevant hit but not
    # a class-specific one" (→ unclassified) vs "nothing at all" (→ no_signal).
    FAMILY_PREFIXES = {
        "COPY": ("ORI_",),
        "VEC":  ("PROM_", "ELEM_"),
        "SP":   ("PROM_", "ELEM_", "ORI_"),
        "BB":   ("AMR_", "ORI_", "PROM_", "ELEM_", "REPORTER_", "TAG_", "REP_"),
    }

    PROBE_CITATIONS = pd.DataFrame([
        # COPY_HIGH
        ("COPY", "HIGH", "ORI_COLE1",
         "ColE1/pMB1 origin; pUC derivatives ~500-700 copies/cell.",
         "Lin-Chao, Chen & Bremer 1992 (PMID 1283002); Addgene Plasmids 101."),
        ("COPY", "HIGH", "ORI_RSF",
         "RSF1010 broad-host-range IncQ origin, >100 copies/cell in E. coli.",
         "Del Solar et al. 1998, Microbiol Mol Biol Rev 62:434."),
        # COPY_LOW
        ("COPY", "LOW", "ORI_P15A",
         "p15A origin, ~10-15 copies/cell.",
         "BioNumbers BNID 100260; Chang & Cohen 1978, J Bacteriol 134:1141."),
        ("COPY", "LOW", "ORI_PSC101",
         "pSC101 origin, ~5 copies/cell.",
         "BioNumbers BNID 103859; Thompson & Cohen 1988, Gene 63:179."),
        # VEC_MAMMALIAN
        ("VEC", "MAMMALIAN", "PROM_CMV",
         "CMV immediate-early promoter; strong constitutive mammalian.",
         "Boshart et al. 1985, Cell 41:521."),
        ("VEC", "MAMMALIAN", "PROM_EF1A",
         "Human EF-1α promoter; constitutive mammalian.",
         "Kim et al. 1990, Gene 91:217."),
        ("VEC", "MAMMALIAN", "PROM_CAG",
         "CAG (CMV-IE enhancer + chicken β-actin + rabbit β-globin intron).",
         "Niwa, Yamamura & Miyazaki 1991, Gene 108:193."),
        ("VEC", "MAMMALIAN", "ELEM_POLYA_BGH",
         "Bovine growth hormone polyA; standard mammalian 3' signal.",
         "Goodwin & Rottman 1992, J Biol Chem 267:16330."),
        ("VEC", "MAMMALIAN", "ELEM_POLYA_SV40",
         "SV40 late polyA; standard mammalian 3' signal.",
         "Fitzgerald & Shenk 1981, Cell 24:251."),
        ("VEC", "MAMMALIAN", "ELEM_CMV_ENHANCER",
         "CMV IE enhancer; mammalian transcriptional activator.",
         "Boshart et al. 1985, Cell 41:521."),
        # VEC_BACTERIAL
        ("VEC", "BACTERIAL", "PROM_T7",
         "T7 promoter; requires T7 RNA polymerase (pET system).",
         "Studier & Moffatt 1986, JMB 189:113."),
        ("VEC", "BACTERIAL", "PROM_T5",
         "T5 bacteriophage promoter; E. coli σ70.",
         "Bujard et al. 1987; pQE-80L manual (Qiagen)."),
        ("VEC", "BACTERIAL", "PROM_LAC",
         "lac/tac/trc promoters; IPTG-inducible E. coli.",
         "de Boer, Comstock & Vasser 1983, PNAS 80:21."),
        # VEC_LENTIVIRAL
        ("VEC", "LENTIVIRAL", "ELEM_LTR_3",
         "3' LTR of HIV-1-derived lentiviral vectors.",
         "Naldini et al. 1996, Science 272:263."),
        ("VEC", "LENTIVIRAL", "ELEM_LTR_5",
         "5' LTR of HIV-1-derived lentiviral vectors.",
         "Naldini et al. 1996, Science 272:263."),
        ("VEC", "LENTIVIRAL", "ELEM_PSI",
         "HIV-1 Ψ packaging signal.",
         "Clever, Sassetti & Parslow 1995, J Virol 69:2101."),
        ("VEC", "LENTIVIRAL", "ELEM_CPPT",
         "Central polypurine tract; improves lentiviral titer.",
         "Follenzi et al. 2000, Nat Genet 25:217."),
        ("VEC", "LENTIVIRAL", "ELEM_WPRE",
         "Woodchuck PRE; boosts transgene expression.",
         "Zufferey et al. 1999, J Virol 73:2886."),
        # VEC_CRISPR
        ("VEC", "CRISPR", "ELEM_GRNA_SCAFFOLD",
         "sgRNA scaffold; canonical CRISPR signature.",
         "Cong et al. 2013, Science 339:819."),
        ("VEC", "CRISPR", "ELEM_TRACRRNA",
         "tracrRNA; canonical CRISPR Type II signature.",
         "Jinek et al. 2012, Science 337:816."),
        ("VEC", "CRISPR", "PROM_U6",
         "Human/mouse U6 Pol III promoter; standard sgRNA driver.",
         "Kunkel & Pederson 1988, NAR 16:7469."),
        # VEC_AAV
        ("VEC", "AAV", "ELEM_AAV_ITR",
         "AAV inverted terminal repeats; required in cis for packaging.",
         "Srivastava et al. 1983, J Virol 45:555."),
        # SP_YEAST
        ("SP", "YEAST", "ORI_2MU",
         "Saccharomyces 2-micron episomal origin.",
         "Broach 1983, Meth Enzymol 101:307."),
        # BB_ entries — cite the canonical Addgene record for each backbone.
        ("BB", "PUC19", "AMR_AMPICILLIN",
         "pUC19: ampicillin + ColE1 + lacZα polylinker.",
         "Addgene #50005; Yanisch-Perron et al. 1985, Gene 33:103."),
        ("BB", "PUC19", "ORI_COLE1",
         "pUC19: ColE1-derived pMB1 high-copy origin.",
         "Addgene #50005; Yanisch-Perron et al. 1985, Gene 33:103."),
        ("BB", "PCDNA3", "PROM_CMV",
         "pcDNA3: CMV promoter + BGH polyA + NeoR + SV40 ori.",
         "Addgene V79020 / Invitrogen pcDNA3 manual."),
        ("BB", "PCDNA3", "ELEM_POLYA_BGH",
         "pcDNA3: BGH polyA downstream of MCS.",
         "Invitrogen pcDNA3 manual."),
        ("BB", "PCDNA3", "AMR_NEOMYCIN",
         "pcDNA3: NeoR (G418/kanamycin dual selection) cassette.",
         "Invitrogen pcDNA3 manual."),
        ("BB", "PCDNA3", "ORI_SV40",
         "pcDNA3: SV40 ori (for episomal replication in T-antigen lines).",
         "Invitrogen pcDNA3 manual."),
        ("BB", "PET28A", "PROM_T7",
         "pET28a: T7 promoter / lac operator.",
         "Novagen pET System Manual (Merck)."),
        ("BB", "PET28A", "AMR_KANAMYCIN",
         "pET28a: KanR selection marker.",
         "Novagen pET System Manual."),
        ("BB", "PET28A", "TAG_HIS",
         "pET28a: N- and C-terminal 6xHis tag cassettes.",
         "Novagen pET System Manual."),
        ("BB", "PEGFP_C1", "REPORTER_EGFP",
         "pEGFP-C1: EGFP under CMV, KanR/NeoR.",
         "Clontech pEGFP-C1 manual (632304)."),
        ("BB", "PEGFP_C1", "PROM_CMV",
         "pEGFP-C1: CMV drives EGFP expression.",
         "Clontech pEGFP-C1 manual."),
        ("BB", "PEGFP_C1", "AMR_KANAMYCIN",
         "pEGFP-C1: KanR/NeoR dual selection.",
         "Clontech pEGFP-C1 manual."),
        ("BB", "PEGFP_N1", "REPORTER_EGFP",
         "pEGFP-N1: C-terminal EGFP fusion under CMV, KanR/NeoR.",
         "Clontech pEGFP-N1 manual (632306)."),
        ("BB", "PEGFP_N1", "PROM_CMV",
         "pEGFP-N1: CMV drives the fusion.",
         "Clontech pEGFP-N1 manual."),
        ("BB", "PEGFP_N1", "AMR_KANAMYCIN",
         "pEGFP-N1: KanR/NeoR dual selection.",
         "Clontech pEGFP-N1 manual."),
        ("BB", "PLENTIGUIDE_PURO", "ELEM_LTR_3",
         "pLentiGuide-Puro: 3' LTR of HIV-1-derived lentiviral vector.",
         "Addgene #52963 (Zhang lab)."),
        ("BB", "PLENTIGUIDE_PURO", "ELEM_LTR_5",
         "pLentiGuide-Puro: 5' LTR of HIV-1-derived lentiviral vector.",
         "Addgene #52963 (Zhang lab)."),
        ("BB", "PLENTIGUIDE_PURO", "AMR_PUROMYCIN",
         "pLentiGuide-Puro: puromycin selection.",
         "Addgene #52963."),
        ("BB", "PLENTIGUIDE_PURO", "PROM_U6",
         "pLentiGuide-Puro: U6 Pol III promoter drives the sgRNA.",
         "Addgene #52963."),
        ("BB", "PLENTIGUIDE_PURO", "ELEM_GRNA_SCAFFOLD",
         "pLentiGuide-Puro: sgRNA scaffold downstream of U6.",
         "Addgene #52963."),
        ("BB", "PDONR221", "AMR_CHLORAMPHENICOL",
         "pDONR221: CmR + ccdB between attP1/attP2 (Gateway entry vector).",
         "Invitrogen Gateway Manual (12535-019)."),
        ("BB", "PDONR223", "AMR_CHLORAMPHENICOL",
         "pDONR223: CmR + ccdB between attP1/attP2 (hORFeome variant).",
         "Addgene #25574 (hORFeome)."),
        ("BB", "PAAV", "ELEM_AAV_ITR",
         "pAAV: ITRs flanking the transgene cassette.",
         "Addgene pAAV-MCS (VPK-410); Srivastava et al. 1983."),
        ("BB", "PAAV2", "ELEM_AAV_ITR",
         "pAAV2: ITRs flanking the transgene cassette (AAV2-serotype).",
         "Addgene pAAV2 variants (e.g. #37825)."),
    ], columns=["soft_cat", "soft_val", "hard_token", "justification", "source"])

    PROBE_CITATIONS
    return FAMILY_PREFIXES, SOFT_TO_HARD, UNCLASSIFIED_FOR


@app.cell
def _(REGISTRY_PATH, pd):
    from collections import defaultdict

    def load_token_sseqid_map(registry_path):
        df = pd.read_parquet(registry_path, columns=["token", "sseqid"])
        if df.empty:
            raise ValueError(f"motif registry is empty: {registry_path}")
        mapping = defaultdict(set)
        for tok, sid in zip(df["token"], df["sseqid"]):
            if not isinstance(tok, str) or not isinstance(sid, str):
                continue
            mapping[tok.strip("<>")].add(sid)
        return dict(mapping)

    token_sseqids = load_token_sseqid_map(REGISTRY_PATH)
    registry_summary = pd.DataFrame(
        sorted((tok, len(sids)) for tok, sids in token_sseqids.items()),
        columns=["hard_token", "n_sseqids"],
    )
    registry_summary.head()
    return (token_sseqids,)


@app.cell
def _(FAMILY_PREFIXES, SOFT_TO_HARD, UNCLASSIFIED_FOR, pd, token_sseqids):
    # Resolve every probe to its sseqid set. Hard tokens missing from the
    # registry are dropped with a loud record so the reviewer can see what was
    # excluded. (Every probe is present today — kept for defensive code.)
    probe_sseqids = {}
    missing_rows = []
    for cat, by_val in SOFT_TO_HARD.items():
        probe_sseqids[cat] = {}
        for val, tokens in by_val.items():
            present = {t for t in tokens if t in token_sseqids}
            absent = tokens - present
            for t in absent:
                missing_rows.append({"soft_cat": cat, "soft_val": val, "hard_token": t})
            probe_sseqids[cat][val] = set().union(*(token_sseqids[t] for t in present))

    uncl_sseqids = {
        cat: set().union(*(token_sseqids.get(t, set()) for t in tokens))
        for cat, tokens in UNCLASSIFIED_FOR.items()
    }

    # Family sseqids: union of all hard tokens in the registry whose name starts
    # with any of the family's prefixes. Used to tell "model produced *something*
    # in the right feature family" from "model produced nothing at all".
    family_sseqids = {}
    for cat, prefixes in FAMILY_PREFIXES.items():
        family_sseqids[cat] = set().union(*(
            sids for tok, sids in token_sseqids.items()
            if any(tok.startswith(p) for p in prefixes)
        ))

    missing_probe_tokens = pd.DataFrame(missing_rows)
    missing_probe_tokens
    return family_sseqids, probe_sseqids, uncl_sseqids


@app.cell
def _(mo):
    mo.md(r"""
    ## Null-buster — does pretraining distinguish `SP_HUMAN` from `SP_ECOLI`?

    If the base model's rate of mammalian-probe hits on `SP_HUMAN` prompts is
    not dramatically higher than on `SP_ECOLI` prompts, pretraining learned
    nothing actionable from `SP_*` and the rest of the notebook is moot. This
    single check decides whether the experiment is worth running.

    Caveat: `SP_HUMAN` prompts almost always co-occur with mammalian hard
    tokens (`PROM_CMV` etc.) — the observed difference is confounded with
    that co-occurrence, which is why we frame it as "does the model produce
    what the prompt asked for" rather than "is SP_ causal".
    """)
    return


@app.cell
def _(base, grpo, probe_sseqids, proportion_confint, tags):
    def _hit_rate(run, tag_mask, probe_set):
        rows_mask = tag_mask.values
        n = int(rows_mask.sum())
        if n == 0:
            return {"k": 0, "n": 0, "rate": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan")}
        hits = sum(
            1 for i, keep in enumerate(rows_mask)
            if keep and run["sseqids_by_row"][i] & probe_set
        )
        lo, hi = proportion_confint(hits, n, alpha=0.05, method="wilson")
        return {"k": hits, "n": n, "rate": hits / n,
                "ci_low": float(lo), "ci_high": float(hi)}

    MAMMAL = probe_sseqids["SP"]["HUMAN"]         # == MAMMALIAN probe set
    BACTER = probe_sseqids["SP"]["ECOLI"]         # bacterial probe set

    _null_rows = []
    for _run_name, _run in [("base", base), ("grpo", grpo)]:
        for _tag_val, _tag_label in [("HUMAN", "SP_HUMAN"), ("ECOLI", "SP_ECOLI")]:
            _tag_mask = tags["sp"] == _tag_val
            for _probe_label, _probe in [("mammalian", MAMMAL), ("bacterial", BACTER)]:
                _null_rows.append({
                    "run": _run_name, "tag": _tag_label, "probe": _probe_label,
                    **_hit_rate(_run, _tag_mask, _probe),
                })
    import pandas as _pd
    null_tbl = _pd.DataFrame(_null_rows)

    # Pull the two base-only numbers the headline claim hinges on.
    _base_mam_on_human = null_tbl[(null_tbl.run == "base") & (null_tbl.tag == "SP_HUMAN")
                                  & (null_tbl.probe == "mammalian")].iloc[0]
    _base_mam_on_ecoli = null_tbl[(null_tbl.run == "base") & (null_tbl.tag == "SP_ECOLI")
                                  & (null_tbl.probe == "mammalian")].iloc[0]
    null_delta = _base_mam_on_human["rate"] - _base_mam_on_ecoli["rate"]
    null_cis_disjoint = (_base_mam_on_human["ci_low"] > _base_mam_on_ecoli["ci_high"]) or \
                        (_base_mam_on_ecoli["ci_low"] > _base_mam_on_human["ci_high"])
    null_tbl
    return null_cis_disjoint, null_delta


@app.cell
def _(mo, null_cis_disjoint, null_delta):
    _verdict = "CLEAN" if null_cis_disjoint and abs(null_delta) > 0.1 else "WEAK or CONFOUNDED"
    mo.md(
        f"""
        **Null-buster verdict: `{_verdict}`.**
        Base `P(mammalian | SP_HUMAN) − P(mammalian | SP_ECOLI) = {null_delta:+.3f}`;
        Wilson CIs disjoint = `{null_cis_disjoint}`.

        If this reads `WEAK or CONFOUNDED`, interpret the SP_ block below
        accordingly — any "conditioning" signal there may be riding on the
        hard-token co-occurrence.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. `COPY_` block — 4-bucket classification

    For each sequence whose prompt had a COPY tag:

    - **correct**: any detected sseqid in the requested class's probe set.
    - **wrong**: any detected sseqid in the opposing class's probe set **and
      none** in the requested class's probe set.
    - **unclassified**: the sequence hit some ORI family sseqid but none in
      either HIGH or LOW probe sets (e.g. only `ORI_F1`, `ORI_SV40`, or
      `ORI_2MU` was detected).
    - **no_signal**: no ORI family sseqid was detected at all.
    """)
    return


@app.cell
def _(base, family_sseqids, grpo, pd, probe_sseqids, tags):
    def _classify_copy(run, tag_col):
        rows = []
        for i, tag in enumerate(tag_col.values):
            if tag is None:
                rows.append(None); continue
            detected = run["sseqids_by_row"][i]
            correct_set = probe_sseqids["COPY"].get(tag, set())
            other_tag = "LOW" if tag == "HIGH" else "HIGH"
            other_set = probe_sseqids["COPY"].get(other_tag, set())
            fam = family_sseqids["COPY"]
            hit_correct = bool(detected & correct_set)
            hit_other = bool(detected & other_set)
            in_family = bool(detected & fam)
            if hit_correct:
                rows.append("correct")
            elif hit_other:
                rows.append("wrong")
            elif in_family:
                rows.append("unclassified")
            else:
                rows.append("no_signal")
        return rows

    _buckets_b = _classify_copy(base, tags["copy"])
    _buckets_g = _classify_copy(grpo, tags["copy"])
    copy_df = pd.DataFrame({"tag": tags["copy"], "base": _buckets_b, "grpo": _buckets_g})
    copy_df = copy_df.dropna(subset=["tag"]).reset_index(drop=True)

    _order = ["correct", "wrong", "unclassified", "no_signal"]
    copy_shares = {}
    for _run_name in ("base", "grpo"):
        _piv = (copy_df.groupby(["tag", _run_name]).size()
                .unstack(fill_value=0).reindex(columns=_order, fill_value=0))
        _piv = _piv.div(_piv.sum(axis=1), axis=0)
        copy_shares[_run_name] = _piv
    copy_shares["base"]
    return copy_df, copy_shares


@app.cell
def _(copy_shares, plt):
    _order = ["correct", "wrong", "unclassified", "no_signal"]
    _colors = {"correct": "#059669", "wrong": "#dc2626",
               "unclassified": "#f59e0b", "no_signal": "#6b7280"}

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for _ax, _run in zip(_axes, ("base", "grpo")):
        _piv = copy_shares[_run]
        _bottoms = None
        for _bucket in _order:
            _vals = _piv[_bucket].values
            _ax.bar(_piv.index, _vals, bottom=_bottoms,
                    color=_colors[_bucket], label=_bucket, edgecolor="white")
            _bottoms = _vals if _bottoms is None else _bottoms + _vals
        _ax.set_title(f"{_run} — COPY_ classification")
        _ax.set_ylabel("fraction of sequences")
        _ax.set_ylim(0, 1)
    _axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(copy_df, mcnemar, pd, proportion_confint):
    # Correct-rate per (run, tag) with Wilson CIs, and paired McNemar on "correct"
    # base-vs-GRPO per tag.
    _rows = []
    for _run_name in ("base", "grpo"):
        for _tag, _grp in copy_df.groupby("tag"):
            _k = int((_grp[_run_name] == "correct").sum())
            _n = len(_grp)
            _lo, _hi = proportion_confint(_k, _n, alpha=0.05, method="wilson")
            _rows.append({
                "run": _run_name, "tag": _tag,
                "correct_rate": _k / _n,
                "ci_low": float(_lo), "ci_high": float(_hi),
                "n": _n,
            })
    copy_wilson = pd.DataFrame(_rows)

    _mc_rows = []
    for _tag, _grp in copy_df.groupby("tag"):
        _b = (_grp["base"] == "correct").to_numpy()
        _g = (_grp["grpo"] == "correct").to_numpy()
        _tab = pd.crosstab(pd.Series(_b, name="base"), pd.Series(_g, name="grpo"))
        _tab = _tab.reindex(index=[False, True], columns=[False, True], fill_value=0)
        _mc = mcnemar(_tab.values, exact=(_tab.values.min() < 25), correction=True)
        _mc_rows.append({"tag": _tag, "n": len(_grp),
                         "delta_correct": _g.mean() - _b.mean(),
                         "mcnemar_p": float(_mc.pvalue)})
    copy_mcnemar = pd.DataFrame(_mc_rows)
    (copy_wilson, copy_mcnemar)
    return


@app.cell
def _(mo):
    mo.md("""
    ### COPY_ — Wilson CIs on correct rate

    "
        f"{copy_wilson.to_markdown(index=False, floatfmt='.3f')}

    "
        "### COPY_ — paired McNemar (base vs GRPO, per tag)

    "
        f"{copy_mcnemar.to_markdown(index=False, floatfmt='.3f')}

    "
        "**Read**: `correct_rate` is the headline. A non-trivial `unclassified`
    "
        "bar means the model produced an ORI but not one on the copy-number
    "
        "axis (often `F1`/`SV40`/`2μ`). GRPO's McNemar p is the sanity check —
    "
        "expected null, since GRPO reward didn't see COPY_.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. `VEC_` block — diagonal heatmap

    For every VEC tag value with at least one probe, compute the rate at which
    sequences with that tag hit each of the five probe sets. The **diagonal**
    is the "model produced features matching the requested vector type"
    rate; off-diagonal is leakage. Tags without clean probes
    (GATEWAY / INSECT / PLANT / RETROVIRAL / REPORTER / YEAST) are listed as a
    count below the heatmap, not plotted as zero.
    """)
    return


@app.cell
def _(base, grpo, np, pd, probe_sseqids, tags):
    vec_probe_vals = list(probe_sseqids["VEC"].keys())
    _all_vec_vals = sorted(tags["vec"].dropna().unique())
    unprobed_vec = [v for v in _all_vec_vals if v not in vec_probe_vals]

    def _hit_matrix(run, tag_col):
        mat = np.zeros((len(vec_probe_vals), len(vec_probe_vals)), dtype=float)
        counts = np.zeros(len(vec_probe_vals), dtype=int)
        for i, tag in enumerate(tag_col.values):
            if tag not in vec_probe_vals:
                continue
            row = vec_probe_vals.index(tag)
            detected = run["sseqids_by_row"][i]
            counts[row] += 1
            for col, val in enumerate(vec_probe_vals):
                if detected & probe_sseqids["VEC"][val]:
                    mat[row, col] += 1
        rates = np.divide(mat, counts[:, None],
                          out=np.zeros_like(mat),
                          where=counts[:, None] > 0)
        return rates, counts

    vec_rates_b, vec_counts_b = _hit_matrix(base, tags["vec"])
    vec_rates_g, _vec_counts_g = _hit_matrix(grpo, tags["vec"])

    unprobed_counts = pd.DataFrame({
        "vec_tag": unprobed_vec,
        "n_prompts": [int((tags["vec"] == v).sum()) for v in unprobed_vec],
    })
    unprobed_counts
    return vec_counts_b, vec_probe_vals, vec_rates_b, vec_rates_g


@app.cell
def _(plt, vec_counts_b, vec_probe_vals, vec_rates_b, vec_rates_g):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for _ax, _mat, _title in [
        (_axes[0], vec_rates_b, "base (n shown per row)"),
        (_axes[1], vec_rates_g, "grpo (n shown per row)"),
    ]:
        _im = _ax.imshow(_mat, vmin=0, vmax=1, cmap="Greens", aspect="auto")
        _ax.set_xticks(range(len(vec_probe_vals)))
        _ax.set_xticklabels(vec_probe_vals, rotation=30, ha="right")
        _ax.set_yticks(range(len(vec_probe_vals)))
        _ax.set_yticklabels([f"{v} (n={vec_counts_b[i]})"
                             for i, v in enumerate(vec_probe_vals)])
        for _i in range(len(vec_probe_vals)):
            for _j in range(len(vec_probe_vals)):
                _ax.text(_j, _i, f"{_mat[_i, _j]:.2f}",
                         ha="center", va="center",
                         color=("white" if _mat[_i, _j] > 0.55 else "black"),
                         fontsize=9)
        _ax.set_title(_title); _ax.set_xlabel("probe set (columns)")
    _axes[0].set_ylabel("VEC_ tag (rows)")
    _fig.colorbar(_im, ax=_axes, shrink=0.8, label="hit rate")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Unprobed VEC_ values (no clean hard-token probe in the registry):**

    "
        f"{unprobed_counts.to_markdown(index=False)}

    "
        "These tags are present in the paired runs but excluded from the "
        "heatmap. Building a clean probe for any of them would require adding "
        "the diagnostic features to the motif registry.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. `SP_` block — target-host group heatmap

    Collapse the 9 species into three probe-able host groups (mammalian,
    bacterial, yeast) and show, for each `SP_*` tag, the rate at which
    sequences hit each group's probe set. `HUMAN/MOUSE/RAT` share the
    mammalian probe — they necessarily tie in this view.

    This section is a sanity check, not a separate finding: `SP_` is largely
    redundant with `VEC_` in the training distribution.
    """)
    return


@app.cell
def _(base, grpo, np, pd, probe_sseqids, tags):
    sp_groups = [
        ("mammalian", probe_sseqids["SP"]["HUMAN"]),
        ("bacterial", probe_sseqids["SP"]["ECOLI"]),
        ("yeast",     probe_sseqids["SP"]["YEAST"]),
    ]
    sp_probed_vals = ["HUMAN", "MOUSE", "RAT", "ECOLI", "YEAST"]
    _all_sp = sorted(tags["sp"].dropna().unique())
    unprobed_sp = [v for v in _all_sp if v not in sp_probed_vals]

    def _sp_matrix(run, tag_col):
        mat = np.zeros((len(sp_probed_vals), len(sp_groups)), dtype=float)
        counts = np.zeros(len(sp_probed_vals), dtype=int)
        for i, tag in enumerate(tag_col.values):
            if tag not in sp_probed_vals:
                continue
            row = sp_probed_vals.index(tag)
            counts[row] += 1
            detected = run["sseqids_by_row"][i]
            for col, (_, probe) in enumerate(sp_groups):
                if detected & probe:
                    mat[row, col] += 1
        return np.divide(mat, counts[:, None],
                         out=np.zeros_like(mat),
                         where=counts[:, None] > 0), counts

    sp_rates_b, sp_counts_b = _sp_matrix(base, tags["sp"])
    sp_rates_g, _ = _sp_matrix(grpo, tags["sp"])

    unprobed_sp_counts = pd.DataFrame({
        "sp_tag": unprobed_sp,
        "n_prompts": [int((tags["sp"] == v).sum()) for v in unprobed_sp],
    })
    unprobed_sp_counts
    return sp_counts_b, sp_groups, sp_probed_vals, sp_rates_b, sp_rates_g


@app.cell
def _(plt, sp_counts_b, sp_groups, sp_probed_vals, sp_rates_b, sp_rates_g):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _group_names = [g[0] for g in sp_groups]
    for _ax, _mat, _title in [
        (_axes[0], sp_rates_b, "base"),
        (_axes[1], sp_rates_g, "grpo"),
    ]:
        _im = _ax.imshow(_mat, vmin=0, vmax=1, cmap="Greens", aspect="auto")
        _ax.set_xticks(range(len(_group_names))); _ax.set_xticklabels(_group_names)
        _ax.set_yticks(range(len(sp_probed_vals)))
        _ax.set_yticklabels([f"{v} (n={sp_counts_b[i]})"
                             for i, v in enumerate(sp_probed_vals)])
        for _i in range(_mat.shape[0]):
            for _j in range(_mat.shape[1]):
                _ax.text(_j, _i, f"{_mat[_i, _j]:.2f}",
                         ha="center", va="center",
                         color=("white" if _mat[_i, _j] > 0.55 else "black"),
                         fontsize=9)
        _ax.set_title(_title); _ax.set_xlabel("probe group")
    _axes[0].set_ylabel("SP_ tag")
    _fig.colorbar(_im, ax=_axes, shrink=0.8, label="hit rate")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    **Unprobed SP_ values (no clean probe: CELEGANS/DROSOPHILA/ZEBRAFISH/"
        "SYNTHETIC):**

    "
        f"{unprobed_sp_counts.to_markdown(index=False)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. `BB_` block — per-backbone completeness

    For each backbone probe, `completeness = |{hard tokens hit}| / |expected|`.
    The "matches this backbone" threshold is `completeness ≥ 0.75` (most of
    the requested canonical tokens must be present — a strict read).
    """)
    return


@app.cell
def _(SOFT_TO_HARD, base, grpo, np, pd, tags, token_sseqids):
    def _completeness(run, tag_col):
        rows = []
        for i, tag in enumerate(tag_col.values):
            if tag is None or tag not in SOFT_TO_HARD["BB"]:
                rows.append({"tag": None, "completeness": np.nan})
                continue
            expected = SOFT_TO_HARD["BB"][tag]
            detected = run["sseqids_by_row"][i]
            hits = sum(
                1 for ht in expected
                if detected & token_sseqids.get(ht, set())
            )
            rows.append({"tag": tag, "completeness": hits / len(expected)})
        return pd.DataFrame(rows)

    bb_b = _completeness(base, tags["bb"]).rename(columns={"completeness": "base"})
    bb_g = _completeness(grpo, tags["bb"]).rename(columns={"completeness": "grpo"})
    assert (bb_b["tag"].fillna("NA") == bb_g["tag"].fillna("NA")).all()
    bb_df = bb_b.assign(grpo=bb_g["grpo"]).dropna(subset=["tag"]).reset_index(drop=True)

    # Per-backbone n, then per-backbone per-run mean completeness + strict-match
    # rate (completeness ≥ 0.75) with Wilson CIs.
    from statsmodels.stats.proportion import proportion_confint as _pci
    _rows = []
    for _tag, _grp in bb_df.groupby("tag"):
        for _run in ("base", "grpo"):
            _vals = _grp[_run].to_numpy()
            _k = int((_vals >= 0.75 - 1e-9).sum())
            _n = len(_vals)
            _lo, _hi = _pci(_k, _n, alpha=0.05, method="wilson")
            _rows.append({
                "backbone": _tag, "run": _run, "n": _n,
                "mean_completeness": float(_vals.mean()),
                "strict_match_rate": _k / _n,
                "ci_low": float(_lo), "ci_high": float(_hi),
            })
    bb_tbl = pd.DataFrame(_rows).sort_values(["backbone", "run"]).reset_index(drop=True)
    bb_tbl
    return (bb_df,)


@app.cell
def _(bb_df, plt):
    _tags_order = sorted(bb_df["tag"].unique())
    _fig, _ax = plt.subplots(figsize=(max(6, len(_tags_order) * 1.0), 4))
    _pos = []
    _labels = []
    for _i, _tag in enumerate(_tags_order):
        _sub = bb_df[bb_df["tag"] == _tag]
        _ax.boxplot([_sub["base"].to_numpy()], positions=[_i * 2.5],
                    widths=0.9, patch_artist=True,
                    boxprops=dict(facecolor="#6b7280", alpha=0.7),
                    medianprops=dict(color="black"))
        _ax.boxplot([_sub["grpo"].to_numpy()], positions=[_i * 2.5 + 1.0],
                    widths=0.9, patch_artist=True,
                    boxprops=dict(facecolor="#059669", alpha=0.7),
                    medianprops=dict(color="black"))
        _pos.append(_i * 2.5 + 0.5); _labels.append(f"{_tag}\n(n={len(_sub)})")
    _ax.set_xticks(_pos); _ax.set_xticklabels(_labels, rotation=30, ha="right", fontsize=8)
    _ax.set_ylabel("completeness (|hit ∩ expected| / |expected|)")
    _ax.set_ylim(-0.05, 1.05)
    _ax.axhline(0.75, color="black", linestyle="--", linewidth=1, alpha=0.5)
    _ax.set_title("Per-backbone completeness — base (grey) vs GRPO (green); dashed = strict threshold 0.75")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Summary table

    One row per (soft category × run): fraction `correct`, fraction
    `unclassified`, fraction `wrong`, and the paired McNemar p on `correct`
    between base and GRPO. For VEC/SP "correct" means diagonal hit
    (requested class's probe set). For BB "correct" means
    `completeness ≥ 0.75`.
    """)
    return


@app.cell
def _(
    SOFT_TO_HARD,
    base,
    family_sseqids,
    grpo,
    probe_sseqids,
    tags,
    token_sseqids,
    uncl_sseqids,
):
    # Generic 4-bucket classifier + BB completeness-threshold variant. Produces
    # per-(run, category) label sequences consumed by the summary table below.
    def bucket_generic(run, tag_col, cat):
        out = []
        fam = family_sseqids[cat]
        uncl = uncl_sseqids.get(cat, set())
        for i, tag in enumerate(tag_col.values):
            if tag is None or tag not in probe_sseqids[cat]:
                out.append(None); continue
            detected = run["sseqids_by_row"][i]
            correct_set = probe_sseqids[cat][tag]
            other_sets = [s for k, s in probe_sseqids[cat].items() if k != tag]
            hit_correct = bool(detected & correct_set)
            hit_other = any(bool(detected & s) for s in other_sets)
            in_family = bool(detected & fam) or bool(detected & uncl)
            if hit_correct:
                out.append("correct")
            elif hit_other:
                out.append("wrong")
            elif in_family:
                out.append("unclassified")
            else:
                out.append("no_signal")
        return out

    def bucket_bb(run, tag_col, threshold=0.75):
        out = []
        fam = family_sseqids["BB"]
        expected_by_bb = SOFT_TO_HARD["BB"]
        for i, tag in enumerate(tag_col.values):
            if tag is None or tag not in expected_by_bb:
                out.append(None); continue
            detected = run["sseqids_by_row"][i]
            def _completeness(bb_name):
                tokens = expected_by_bb[bb_name]
                hits = sum(1 for ht in tokens if detected & token_sseqids.get(ht, set()))
                return hits / len(tokens)
            c_self = _completeness(tag)
            c_others = [_completeness(k) for k in expected_by_bb if k != tag]
            in_family = bool(detected & fam)
            if c_self >= threshold - 1e-9:
                out.append("correct")
            elif any(c >= threshold - 1e-9 for c in c_others):
                out.append("wrong")
            elif in_family:
                out.append("unclassified")
            else:
                out.append("no_signal")
        return out

    # Compute buckets for all four categories, both runs.
    bucket_cols = {}
    for _run_name, _run in [("base", base), ("grpo", grpo)]:
        bucket_cols[(_run_name, "COPY")] = bucket_generic(_run, tags["copy"], "COPY")
        bucket_cols[(_run_name, "VEC")]  = bucket_generic(_run, tags["vec"],  "VEC")
        bucket_cols[(_run_name, "SP")]   = bucket_generic(_run, tags["sp"],   "SP")
        bucket_cols[(_run_name, "BB")]   = bucket_bb(_run, tags["bb"])
    return (bucket_cols,)


@app.cell
def _(bucket_cols, mcnemar, pd, proportion_confint):
    _rows = []
    _cats = ("COPY", "VEC", "SP", "BB")
    for _cat in _cats:
        _b = pd.Series(bucket_cols[("base", _cat)])
        _g = pd.Series(bucket_cols[("grpo", _cat)])
        _mask = _b.notna() & _g.notna()
        _b = _b[_mask].reset_index(drop=True)
        _g = _g[_mask].reset_index(drop=True)
        _n = len(_b)
        if _n == 0:
            continue
        _bc = (_b == "correct").to_numpy()
        _gc = (_g == "correct").to_numpy()
        _blo, _bhi = proportion_confint(int(_bc.sum()), _n, alpha=0.05, method="wilson")
        _glo, _ghi = proportion_confint(int(_gc.sum()), _n, alpha=0.05, method="wilson")
        _tab = pd.crosstab(pd.Series(_bc, name="b"), pd.Series(_gc, name="g"))
        _tab = _tab.reindex(index=[False, True], columns=[False, True], fill_value=0)
        _mc = mcnemar(_tab.values, exact=(_tab.values.min() < 25), correction=True)
        _rows.append({
            "category": _cat, "n": _n,
            "base_correct": _bc.mean(),
            "base_ci_low": float(_blo), "base_ci_high": float(_bhi),
            "grpo_correct": _gc.mean(),
            "grpo_ci_low": float(_glo), "grpo_ci_high": float(_ghi),
            "delta":       _gc.mean() - _bc.mean(),
            "mcnemar_p":   float(_mc.pvalue),
            "base_unclassified": (_b == "unclassified").mean(),
            "grpo_unclassified": (_g == "unclassified").mean(),
            "base_wrong": (_b == "wrong").mean(),
            "grpo_wrong": (_g == "wrong").mean(),
            "base_no_signal": (_b == "no_signal").mean(),
            "grpo_no_signal": (_g == "no_signal").mean(),
        })
    summary_tbl = pd.DataFrame(_rows)
    summary_tbl
    return


@app.cell
def _(mo):
    mo.md("""
    ### Paired summary (all four soft categories)

    "
        f"{summary_tbl.to_markdown(index=False, floatfmt='.3f')}

    "
        "**How to read.**
    "
        "- `base_correct` is the headline — does the pretrained model produce
    "
        "  features matching the requested soft value?
    "
        "- `mcnemar_p` should be ≥ ~0.05 for all four categories (null:
    "
        "  GRPO reward didn't see these tokens, so no shift is expected).
    "
        "- `unclassified` is the honest epistemic bucket — do *not* fold it
    "
        "  into either success or failure.
    "
        "- `wrong` is small when the model is conditioning; large `wrong` +
    "
        "  small `correct` is the signature of *ignoring* the soft tag.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
