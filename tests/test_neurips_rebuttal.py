"""Focused tests for the final rebuttal analysis helpers."""

import json
import sys
from pathlib import Path

import pytest

REBUTTAL_DIR = Path(__file__).resolve().parents[1] / "experiments" / "neurips_rebuttal"
if str(REBUTTAL_DIR) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_DIR))

import baselines  # noqa: E402
import build_benchmark  # noqa: E402
import common  # noqa: E402
import exhaustive_training_mosaic as mosaic  # noqa: E402
import heldout_registry_multiseed as heldout  # noqa: E402
import summarize_exhaustive_blast_thresholds as thresholds  # noqa: E402


def test_cluster_panel_keeps_separate_independent_minority_units(tmp_path):
    import pandas as pd
    from pretraining.build_cluster_split import sha256_file

    split = tmp_path / "split"
    (split / "audit").mkdir(parents=True)
    rows = pd.DataFrame([
        dict(cluster_id="a", sequence_id="s1", plasmid_id="p1", sequence="ACGT",
             prompt="<BOS><ORI_P15A><AMR_CHLORAMPHENICOL><SEQ>"),
        dict(cluster_id="a", sequence_id="s1", plasmid_id="p1", sequence="ACGT",
             prompt="<BOS><ORI_P15A><SEQ>"),
        dict(cluster_id="a", sequence_id="s2", plasmid_id="p2", sequence="ACGA",
             prompt="<BOS><ORI_P15A><COPY_LOW><SEQ>"),
        dict(cluster_id="b", sequence_id="s3", plasmid_id="p3", sequence="ACGG",
             prompt="<BOS><ORI_P15A><ORI_PSC101><SEQ>"),
        dict(cluster_id="c", sequence_id="s4", plasmid_id="p4", sequence="ACGC",
             prompt="<BOS><ORI_COLE1><SEQ>"),
    ])
    hashes = {}
    for part in ("train", "validation", "test"):
        path = split / f"{part}.parquet"
        rows.to_parquet(path, index=False)
        hashes[part] = sha256_file(path)
    audit = split / "audit/report.json"
    audit.write_text(json.dumps(dict(passed=True, partition_sha256=hashes)))
    output = tmp_path / "panel.parquet"
    report = build_benchmark.build_cluster_panel(split, output, n=2, minority_n=10)
    panel = pd.read_parquet(output)
    assert panel.sample_id.is_unique
    assert panel.sequence_id.is_unique
    assert panel[panel.panel_main].cluster_id.nunique() == 2
    assert len(panel[panel.panel_main]) == 2
    assert report["minority_panels"]["<ORI_P15A>"]["selected_clusters"] == 2
    assert report["minority_panels"]["<ORI_PSC101>"]["below_target"]
    for token in report["minority_panels"]:
        subset = panel[panel.minority_panels.map(lambda labels: token in labels)]
        assert subset.cluster_id.is_unique
        assert subset.prompt.str.contains(token, regex=False).all()
    second = tmp_path / "second.parquet"
    build_benchmark.build_cluster_panel(split, second, n=2, minority_n=10)
    pd.testing.assert_frame_equal(panel, pd.read_parquet(second))
    with pytest.raises(FileExistsError):
        build_benchmark.build_cluster_panel(split, output)
    audit.write_text(json.dumps(dict(passed=False, partition_sha256=hashes)))
    with pytest.raises(ValueError, match="did not pass"):
        build_benchmark.build_cluster_panel(split, tmp_path / "failed.parquet")
    audit.write_text(json.dumps(dict(passed=True, partition_sha256={**hashes, "test": "bad"})))
    with pytest.raises(ValueError, match="does not match test"):
        build_benchmark.build_cluster_panel(split, tmp_path / "mismatch.parquet")


def test_load_annotations_rejects_duplicate_sequence_ids(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            [
                {"seq_id": "gen_0", "features": []},
                {"seq_id": "gen_0", "features": []},
            ]
        )
    )

    with pytest.raises(ValueError, match="Duplicate seq_id"):
        common.load_annotations(path)


def test_paired_statistics_reject_misaligned_panels():
    with pytest.raises(ValueError, match="differ in length"):
        common.mcnemar_exact([True], [True, False])
    with pytest.raises(ValueError, match="differ in length"):
        common.paired_bootstrap_ci([True], [True, False])


def test_mosaic_parser_keeps_queries_without_hits(tmp_path):
    blast = tmp_path / "panel.tsv"
    blast.write_text("q0\ttrain_0\t99\t4\t4\t4\t1\t4\t8\t1e-10\n")

    coverage, diagnostics = mosaic.parse_mosaic(blast, {"q0": 4, "q1": 4})

    assert coverage == {"q0": 1.0, "q1": 0.0}
    assert diagnostics["queries_with_hits"] == 1


def test_threshold_parser_keeps_fasta_queries_without_hits(tmp_path):
    fasta = tmp_path / "panel.fasta"
    blast = tmp_path / "panel.tsv"
    fasta.write_text(">q0\nAAAA\n>q1\nCCCC\n")
    blast.write_text("q0\ttrain_0\t99\t4\t4\t4\t1\t4\t8\t1e-10\n")

    rows = thresholds.parse_panel("panel", fasta, blast)
    by_id = {row["query_id"]: row for row in rows}

    assert by_id["q0"]["union_pident_ge_90"] == 1.0
    assert by_id["q1"]["union_pident_ge_90"] == 0.0
    assert by_id["q1"]["best_subject_pident_ge_90"] == 0.0


def test_seed_paths_reject_duplicates():
    with pytest.raises(ValueError, match="only once"):
        heldout.parse_seed_paths(["42=first.json", "42=second.json"])


def test_exact_heldout_detection_applies_quality_filter():
    features = [
        {"sseqid": "Kept", "percmatch": 95.0, "fragment": False},
        {"sseqid": "low", "percmatch": 94.9, "fragment": False},
        {"sseqid": "fragment", "percmatch": 100.0, "fragment": True},
    ]

    assert heldout.detected_sseqids(features) == {"kept"}


def test_released_grpo_config_pins_historical_reward_cap():
    from post_training.configs.grpo_plannotate_anyscale import config as released
    from post_training.configs.grpo_plannotate_v3_anyscale import config as v3

    assert released.scorer_kwargs["norm_score_cap"] == 5.0
    assert v3.scorer_kwargs["norm_score_cap"] == 1.5


def test_plannotate_scorer_uses_configured_reward_cap():
    from post_training.scorers.plannotate import PlannotateScorer

    scorer = PlannotateScorer.__new__(PlannotateScorer)
    scorer.norm_score_cap = 5.0
    scorer._run_blast = lambda sequence, tokens: {
        "<AMR_KANAMYCIN>": {
            "pct_id": 100.0,
            "coverage": 100.0,
            "norm_score": 3.0,
            "bit_score": 300.0,
            "evalue": 0.0,
            "sseqid": "KanR",
        }
    }

    result = scorer.score_sequence_detailed("<AMR_KANAMYCIN>", "A" * 30)

    assert result["reward"] == pytest.approx(0.81)


def test_retrieval_baseline_accepts_preselected_ids(tmp_path, capsys):
    import types
    import pandas as pd

    bench = tmp_path / "bench.parquet"
    train = tmp_path / "train.parquet"
    ids_file = tmp_path / "ids.txt"
    ann_file = tmp_path / "ann.json"

    pd.DataFrame({"prompt": ["<ORI_COLE1> <AMR_AMPICILLIN>"]}).to_parquet(bench)
    pd.DataFrame({"plasmid_id": [1], "prompt": ["<ORI_COLE1>"]}).to_parquet(train)
    ids_file.write_text("24552\n")
    ann_file.write_text(
        json.dumps(
            [
                {
                    "seq_id": "24552",
                    "features": [
                        {"Feature": "ColE1", "Type": "rep_origin", "percmatch": 99.0, "fragment": False},
                        {"Feature": "AmpR", "Type": "CDS", "percmatch": 99.0, "fragment": False},
                    ],
                }
            ]
        )
    )

    args = types.SimpleNamespace(
        train_pairs=train,
        benchmark=bench,
        annotations=ann_file,
        retrieved_ids=ids_file,
        out=None,
    )
    baselines.cmd_retrieval(args)
    out = capsys.readouterr().out
    assert "viability                : 100.0%" in out
    assert "useful                   : 100.0%" in out
