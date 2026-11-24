"""Unit tests for canonical manuscript figure and auxiliary diagnostic scripts."""

from __future__ import annotations

import math

import pandas as pd

from eval.scripts import generate_figures
from experiments.neurips_rebuttal import manuscript_diagnostics as md


def test_generate_figures_cli_and_default_paths(tmp_path):
    rigor_dir, registry, out_dir = generate_figures.resolve_default_paths()
    assert rigor_dir.name == "rigor_20261116"
    assert registry.name == "motif_registry_combined.parquet"
    assert out_dir.name == "paper"

    custom = tmp_path / "figs"
    parsed = generate_figures.parse_args(["--out-dir", str(custom)])
    assert parsed.out_dir == custom
    assert parsed.rigor_dir == rigor_dir


def test_manuscript_diagnostics_helpers_and_cli(tmp_path):
    stats = md._seed_mean_std([0.80, 0.81, 0.82])
    assert math.isclose(stats["mean"], 0.81, rel_tol=1e-9)
    assert math.isclose(stats["std"], 0.01, rel_tol=1e-9)

    single = md._seed_mean_std([0.50])
    assert single["std"] == 0.0

    # Clean-prompt sensitivity excludes noisy ELEM_IRES/TRACRRNA tokens cleanly.
    panel = pd.DataFrame([
        {
            "sample_id": "s1",
            "cluster_id": "c1",
            "prompt": "<BOS><AMR_AMPICILLIN><ELEM_IRES><ELEM_TRACRRNA><SEQ>",
        },
        {
            "sample_id": "s2",
            "cluster_id": "c2",
            "prompt": "<BOS><ORI_COLE1><PROM_T7><SEQ>",
        },
    ])
    obs_base_one = pd.DataFrame([
        {
            "sample_id": "s1",
            "cluster_id": "c1",
            "prompt": panel.loc[0, "prompt"],
            "training_seed": 42,
            "viable": 1.0,
            "detected_tokens_json": '["<AMR_AMPICILLIN>"]',
        },
        {
            "sample_id": "s2",
            "cluster_id": "c2",
            "prompt": panel.loc[1, "prompt"],
            "training_seed": 42,
            "viable": 1.0,
            "detected_tokens_json": '["<ORI_COLE1>", "<PROM_T7>"]',
        },
    ])
    obs_base = pd.concat(
        [obs_base_one.assign(seed=d) for d in (42, 43, 44, 45)],
        ignore_index=True,
    )
    obs_grpo = pd.concat(
        [obs_base.assign(training_seed=s) for s in (42, 43, 44)],
        ignore_index=True,
    )
    res = md.compute_clean_prompt_sensitivity(panel, obs_base, obs_base, obs_grpo, iters=20)
    assert res["reference_addgene"]["faithfulness_clean"] == 1.0
    assert res["reference_addgene"]["useful_10_clean"] == 1.0

    parsed = md.parse_args(["--out-dir", str(tmp_path), "--run-blast", "--workers", "2"])
    assert parsed.out_dir == tmp_path
    assert parsed.run_blast is True
    assert parsed.workers == 2
