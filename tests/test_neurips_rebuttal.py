"""Focused tests for the final rebuttal analysis helpers."""

import json
import sys
from pathlib import Path

import pytest

REBUTTAL_DIR = Path(__file__).resolve().parents[1] / "experiments" / "neurips_rebuttal"
if str(REBUTTAL_DIR) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_DIR))

import common  # noqa: E402
import exhaustive_training_mosaic as mosaic  # noqa: E402
import heldout_registry_multiseed as heldout  # noqa: E402
import summarize_exhaustive_blast_thresholds as thresholds  # noqa: E402


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
