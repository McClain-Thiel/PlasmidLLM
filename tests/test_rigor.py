"""Regression tests for the reward ablation and cluster-clean data path."""

import random
import shutil

import pandas as pd
import pytest
import torch
from Bio.Seq import Seq

from plasmid_llm.data import PlasmidDataset, minority_sampling_weights, validate_disjoint
from post_training.scorers.plannotate_v2 import (
    PlannotateV2Scorer, complete_orfs, reward_components,
)
from pretraining.build_cluster_split import assign_partitions, canonical_circle
from pretraining.verify_cluster_split import circular_intervals, interval_union, violations_from_blast


def test_reward_penalizes_extras_and_incomplete_orfs():
    expected = {"<AMR_A>", "<ORI_A>"}
    clean = reward_components(expected, expected, {"<AMR_A>"})
    extra = reward_components(expected, expected | {"<ORI_B>"}, {"<AMR_A>"})
    broken = reward_components(expected, expected, set())
    assert clean["reward"] == pytest.approx(1)
    assert extra["reward"] < clean["reward"]
    assert extra["extra_ori_amr"] == 1
    assert broken["reward"] < clean["reward"]
    assert reward_components(set(), {"<ORI_A>"}, set())["reward"] == 0
    assert reward_components({"<ORI_A>"}, set(), set())["reward"] == 0


@pytest.fixture
def coding_dna():
    rng = random.Random(9)
    codons = ["GCT", "AAA", "GAC", "TTC", "GGA", "ACT", "TAT", "GTT", "CTG", "CAG"]
    return "ATG" + "".join(rng.choice(codons) for _ in range(119)) + "TAA"


def test_complete_orfs_are_circular_and_strand_invariant(coding_dna):
    expected = str(Seq(coding_dna).translate(table=11, cds=True))
    for dna in (coding_dna, coding_dna[17:] + coding_dna[:17],
                str(Seq(coding_dna).reverse_complement())):
        assert expected in complete_orfs(dna)
    assert expected not in complete_orfs(coding_dna[:60] + "TGA" + coding_dna[63:])
    assert expected not in complete_orfs(coding_dna[:60] + "N" + coding_dna[61:])
    assert complete_orfs("ATG" + "GCT" * 60) == []


@pytest.fixture
def v2(tmp_path, coding_dna):
    if any(shutil.which(program) is None for program in ("makeblastdb", "blastn", "blastp", "blastx")):
        pytest.skip("NCBI BLAST+ is not installed")
    rng = random.Random(31)
    ori = "".join(rng.choice("ACGT") for _ in range(240))
    registry = pd.DataFrame([
        dict(token="<AMR_TEST>", sseqid="gene", seq_type="dna", sequence=coding_dna),
        dict(token="<ORI_TEST>", sseqid="ori", seq_type="dna", sequence=ori),
        dict(token="<TAG_HIS>", sseqid="6xHis", seq_type="dna", sequence="CAT" * 6),
    ])
    return PlannotateV2Scorer(motif_registry_df=registry, db_dir=str(tmp_path), num_threads=1), ori


def test_real_blast_reward_detects_frameshifts_and_extra_origins(v2, coding_dna):
    scorer, ori = v2
    prompt = "<AMR_TEST>"
    clean, extra, frameshift, stop = scorer.score_batch_detailed(
        [prompt] * 4,
        [coding_dna, coding_dna + ori, coding_dna[:90] + coding_dna[91:],
         coding_dna[:90] + "TAA" + coding_dna[93:]],
    )
    assert clean["intact_orf_recall"] == 1
    assert extra["extra_ori_amr"] == 1
    assert extra["reward"] < clean["reward"]
    assert frameshift["intact_orf_recall"] == 0
    assert stop["intact_orf_recall"] == 0


def test_synonymous_peptide_tag_and_batch_invariance(v2, coding_dna):
    scorer, _ = v2
    dna = coding_dna[:-3] + "CAC" * 6 + "TAA"
    prompt = "<AMR_TEST><TAG_HIS>"
    result = scorer.score_sequence_detailed(prompt, dna)
    assert result["peptide_tag_recall"] == 1
    assert scorer.score_sequence(prompt, dna[19:] + dna[:19]) == result["reward"]
    assert scorer.score_batch([prompt, "<ORI_TEST>"], [dna, ""])[0] == result["reward"]
    with pytest.raises(ValueError, match="counts differ"):
        scorer.score_batch([prompt], [])
    with pytest.raises(ValueError, match="No validated"):
        scorer.score_sequence("<AMR_UNKNOWN>", dna)


def test_promoter_only_amr_cannot_earn_gene_credit(tmp_path):
    registry = pd.DataFrame([dict(token="<AMR_TEST>", sseqid="cat_promoter", seq_type="dna",
                                  sequence="ACGT" * 100)])
    scorer = PlannotateV2Scorer(motif_registry_df=registry, db_dir=str(tmp_path))
    with pytest.raises(ValueError, match="No validated"):
        scorer.score_sequence("<AMR_TEST>", "ACGT" * 100)


def test_circular_canonicalization_matches_brute_force():
    rng = random.Random(43)
    for n in range(1, 50):
        dna = "".join(rng.choice("ACGTN") for _ in range(n))
        rc = str(Seq(dna).reverse_complement())
        expected = min(s[i:] + s[:i] for s in (dna, rc) for i in range(n))
        assert canonical_circle(dna) == expected


def test_cluster_split_keeps_source_versions_and_duplicates_together():
    rows = pd.DataFrame(dict(plasmid_id=[f"p{i}" for i in range(100)],
                             sequence_id=[f"s{i}" for i in range(100)]))
    rows.loc[1, "plasmid_id"] = "p0"
    clusters = pd.DataFrame(dict(representative=rows.sequence_id, member=rows.sequence_id))
    clusters.loc[2, "representative"] = "s1"
    split = assign_partitions(rows, clusters)
    assert split.iloc[:3].cluster_id.nunique() == 1
    assert split.iloc[:3].partition.nunique() == 1
    pd.testing.assert_frame_equal(split, assign_partitions(rows, clusters.sample(frac=1)))
    with pytest.raises(ValueError, match="complete input"):
        assign_partitions(rows, clusters.iloc[:-1])


class RecordingTokenizer:
    pad_token_id, bos_token_id, sep_token_id, eos_token_id = 0, 1, 2, 3

    def __init__(self):
        self.encoded = []

    def encode(self, text):
        self.encoded.append(text)
        return [4] * len(text)


def test_training_augmentation_preserves_hard_tokens_and_dna(tmp_path):
    path = tmp_path / "pairs.parquet"
    pd.DataFrame([dict(prompt="<SP_ECOLI><VEC_BACTERIAL><COPY_LOW><ORI_P15A><AMR_A>",
                       sequence="ACGTAGC", plasmid_id="a", sequence_id="s", cluster_id="c")]).to_parquet(path)
    tokenizer = RecordingTokenizer()
    ds = PlasmidDataset(str(path), tokenizer, 128, circular_rotation=True, soft_token_dropout=1)
    torch.manual_seed(42)
    sample = ds[0]
    assert tokenizer.encoded[0] == "<COPY_LOW><ORI_P15A><AMR_A>"
    assert canonical_circle(tokenizer.encoded[1]) == canonical_circle("ACGTAGC")
    assert sample["input_ids"].shape == sample["labels"].shape == (128,)
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint(ds, ds)


def test_minority_weights_are_capped_and_fail_on_absent_classes():
    rare = "<ORI_P15A><ORI_PSC101><AMR_CHLORAMPHENICOL><COPY_LOW>"
    weights = minority_sampling_weights([rare] + ["<ORI_COLE1>"] * 999)
    assert weights[0] == 4
    assert weights[1:].eq(1).all()
    with pytest.raises(ValueError, match="no examples"):
        minority_sampling_weights(["<ORI_COLE1>"])


def test_leakage_audit_counts_circular_hsps_without_double_counting(tmp_path):
    assert circular_intervals(91, 120, 100) == [(90, 100), (0, 20)]
    assert interval_union([(0, 50), (25, 75)]) == 75
    hits = tmp_path / "hits.tsv"
    hits.write_text("q\ts\t95\t1\t95\t1\t95\t200\t100\n"
                    "q\ts\t95\t101\t195\t1\t95\t200\t100\n"
                    "q2\ts\t95\t1\t60\t1\t60\t200\t100\n"
                    "q2\ts\t95\t101\t160\t1\t60\t200\t100\n")
    result = violations_from_blast(hits, 0.9, 0.9)
    assert len(result) == 1
    assert result[0]["query_coverage"] == 0.95


def test_anyscale_data_discovery_handles_numeric_reward_settings(tmp_path, monkeypatch):
    from infra.anyscale import run_anyscale
    from post_training.config import PostTrainingConfig

    monkeypatch.setattr(run_anyscale, "DATA_DIR", tmp_path)
    path = tmp_path / "split" / "train.parquet"
    config = PostTrainingConfig(prompts_parquet=str(path),
                                scorer_kwargs={"min_identity": 90.0, "filter_db": True})
    assert run_anyscale.required_data_paths(config) == [path]


def test_ablation_configs_differ_only_in_reward_and_run_identity(monkeypatch):
    from post_training.runners.run import load_config

    for key, value in {
        "PLASMID_RUN_ID": "unit-test", "PLASMID_BASE_MODEL": "owner/model",
        "PLASMID_BASE_REVISION": "a" * 40, "PLASMID_TRAIN_PROMPTS": "data/train.parquet",
        "PLASMID_VALIDATION_PROMPTS": "data/validation.parquet",
        "S3_CHECKPOINT_PREFIX": "s3://unit-test/checkpoints",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLASMID_REWARD_VERSION", "v1")
    v1 = load_config("post_training/configs/grpo_reward_ablation.py")
    monkeypatch.setenv("PLASMID_REWARD_VERSION", "v2")
    v2 = load_config("post_training/configs/grpo_reward_ablation.py")
    differing = {k for k, value in vars(v1).items() if value != getattr(v2, k)}
    assert differing == {"scorer", "scorer_kwargs", "checkpoint_dir", "s3_checkpoint_prefix",
                         "wandb_run_name", "wandb_tags"}
    assert v1.max_new_tokens == v2.max_new_tokens == 4096


def test_orthogonal_intact_endpoint_rejects_partial_and_ambiguous_calls(tmp_path):
    from experiments.neurips_rebuttal.orthogonal import complete_protein_ids, is_intact_hit

    proteins = tmp_path / "proteins.faa"
    proteins.write_text(">gen_0_1 # 1 # 90 # 1 # ID=1;partial=00;start_type=ATG\nMAKKL\n"
                        ">gen_1_1 # 1 # 90 # 1 # ID=2;partial=10;start_type=Edge\nMAKKL\n"
                        ">gen_2_1 # 1 # 90 # 1 # ID=3;partial=00;start_type=ATG\nMAXKL\n")
    complete = complete_protein_ids(proteins)
    assert complete == {"gen_0_1"}
    row = {"Protein id": "gen_0_1", "Method": "EXACT", "% Coverage of reference protein": "100"}
    assert is_intact_hit(row, complete)
    assert not is_intact_hit({**row, "Method": "PARTIAL"}, complete)
    assert not is_intact_hit({**row, "% Coverage of reference protein": "89"}, complete)
    assert not is_intact_hit({**row, "Protein id": "gen_1_1"}, complete)
    with pytest.raises(ValueError, match="missing columns"):
        is_intact_hit({"Protein id": "gen_0_1"}, complete)


def test_cluster_bootstrap_preserves_pairing_and_rejects_missing_units():
    from experiments.neurips_rebuttal.common import paired_cluster_bootstrap_ci

    base = [[0, 0, 0], [1, 0, 0]]
    treatment = [[0.25, 0.25, 0.25], [1.25, 0.25, 0.25]]
    assert paired_cluster_bootstrap_ci(base, treatment, ["a", "a", "b"], iters=100) == (0.25, 0.25)
    with pytest.raises(ValueError, match="two source clusters"):
        paired_cluster_bootstrap_ci(base, treatment, ["a"] * 3)
    with pytest.raises(ValueError, match="matched nonempty"):
        paired_cluster_bootstrap_ci(base, treatment[:1], ["a", "a", "b"])


def test_rebalanced_trainer_batches_support_finite_gradients(tmp_path):
    pytest.importorskip("accelerate")
    from transformers import TrainingArguments
    from pretraining.train import RebalancedTrainer
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMConfig, PlasmidLMForCausalLM

    config = PlasmidLMConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32)
    model = PlasmidLMForCausalLM(config)
    ids = torch.arange(4, 20, dtype=torch.long)
    dataset = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
               for _ in range(4)]
    trainer = RebalancedTrainer(
        model=model, train_dataset=dataset, sampling_weights=torch.tensor([1., 1., 4., 1.]),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
                               per_device_train_batch_size=2, dataloader_pin_memory=False),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["input_ids"].shape == (2, 16)
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_evaluation_decodes_only_completion_and_stops_at_eos():
    from eval.scripts.generate import completion_record

    class Tokenizer:
        eos_token_id = 1

        def decode(self, ids, skip_special_tokens):
            assert not skip_special_tokens
            tokens = {0: "<PAD>", 1: "<EOS>", 2: "<SEQ>", 3: "<SEP>",
                      4: "ACG", 5: "TTA", 6: "<AMR_A>"}
            return "".join(tokens[i] for i in ids)

    # A generated separator must not discard the DNA preceding it, and neither
    # the prompt nor the padding after EOS belongs in sequence-length metrics.
    row = completion_record(Tokenizer(), [0, 6, 2, 4, 3, 5, 1, 0, 0], 3, 6)
    assert row["sequence"] == "ACGTTA"
    assert row["length"] == 6
    assert row["generated_tokens"] == 4
    assert row["has_eos"] and not row["truncated"]
    capped = completion_record(Tokenizer(), [6, 2, 4, 5], 2, 2)
    assert capped["truncated"] and not capped["has_eos"]


def test_pretraining_selection_requires_consistent_early_stopping(tmp_path):
    from plasmid_llm.config import PretrainingConfig

    data = tmp_path / "data.parquet"
    data.touch()
    tokens = tmp_path / "tokens.txt"
    tokens.touch()
    with pytest.raises(ValueError, match="best-model loading"):
        PretrainingConfig(data, tokens, load_best_model_at_end=False)
    config = PretrainingConfig(data, tokens, load_best_model_at_end=False,
                               early_stopping_patience=None)
    assert config.to_mlflow_params()["load_best_model_at_end"] is False


@pytest.mark.parametrize("kind", ["kmer", "char"])
def test_fresh_pretraining_tokenizer_auto_reload(tmp_path, kind, monkeypatch):
    from transformers import AutoTokenizer
    from transformers import dynamic_module_utils
    from plasmid_llm.config import PretrainingConfig
    from pretraining.train import build_tokenizer

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    data = tmp_path / "data.parquet"
    data.touch()
    tokens = tmp_path / "special_tokens.txt"
    tokens.write_text("<BOS>\n<EOS>\n<SEP>\n<PAD>\n<UNK>\n<SEQ>\n")
    config = PretrainingConfig(data, tokens, tokenizer_type=kind, output_dir=tmp_path / "model")
    tokenizer = build_tokenizer(config)
    tokenizer.save_pretrained(config.output_dir)
    restored = AutoTokenizer.from_pretrained(config.output_dir, trust_remote_code=True,
                                             local_files_only=True)
    text = "<BOS><SEQ>ACGTACGTACGT<EOS>"
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.pad_token_id == tokenizer.pad_token_id
    assert restored.eos_token_id == tokenizer.eos_token_id


def test_trainer_accumulation_matches_single_batch_gradients(tmp_path):
    import copy
    from transformers import Trainer, TrainingArguments
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMConfig, PlasmidLMForCausalLM

    torch.manual_seed(7)
    model = PlasmidLMForCausalLM(PlasmidLMConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32))
    accumulated = copy.deepcopy(model)
    ids = torch.randint(4, 32, (4, 12))
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    losses, gradients = [], []
    for current, accumulation in ((model, 1), (accumulated, 2)):
        trainer = Trainer(model=current, args=TrainingArguments(
            output_dir=str(tmp_path / str(accumulation)), use_cpu=True, report_to=[],
            gradient_accumulation_steps=accumulation))
        trainer.current_gradient_accumulation_steps = accumulation
        loss = torch.tensor(0.)
        for start in range(0, 4, 4 // accumulation):
            microbatch = {k: v[start:start + 4 // accumulation] for k, v in batch.items()}
            loss += trainer.training_step(current, microbatch, num_items_in_batch=torch.tensor(44))
        losses.append(loss)
        gradients.append({name: p.grad.clone() for name, p in current.named_parameters()
                          if p.grad is not None})
    torch.testing.assert_close(losses[0], losses[1])
    for name, expected in gradients[0].items():
        torch.testing.assert_close(gradients[1][name], expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("corrupt", [False, True])
def test_pretraining_export_publishes_manifest_only_after_readback(tmp_path, corrupt):
    import io
    import json
    from infra.anyscale.run_cluster_pretraining import export_final_model

    final = tmp_path / "final"
    final.mkdir()
    for name in ("model.safetensors", "config.json", "vocab.json", "tokenizer_config.json"):
        (final / name).write_bytes(b"checkpoint contents")
    (final / "training_report.json").write_text(json.dumps(
        dict(selection="last_step", global_step=200, requested_steps=200)))

    class Body(io.BytesIO):
        def iter_chunks(self, chunk_size):
            while chunk := self.read(chunk_size):
                yield chunk

    class Client:
        def __init__(self):
            self.objects = {}

        def put_object(self, **kwargs):
            key = kwargs["Key"]
            assert kwargs["IfNoneMatch"] == "*"
            if key in self.objects:
                raise FileExistsError(key)
            value = kwargs["Body"]
            self.objects[key] = value.read() if hasattr(value, "read") else value

        def get_object(self, **kwargs):
            return {"Body": Body(b"corrupted" if corrupt else self.objects[kwargs["Key"]])}

    client = Client()
    if corrupt:
        with pytest.raises(ValueError, match="readback checksum"):
            export_final_model(client, final, "s3://bucket/fresh")
        assert "fresh/export_manifest.json" not in client.objects
        assert not (tmp_path / "export_manifest.json").exists()
    else:
        report = export_final_model(client, final, "s3://bucket/fresh")
        assert len(report["files"]) == 5
        assert json.loads(client.objects["fresh/export_manifest.json"]) == report
        with pytest.raises(FileExistsError):
            export_final_model(client, final, "s3://bucket/fresh")


def test_run_sequence_stops_after_failure(tmp_path):
    import json
    import subprocess
    import sys
    from infra.anyscale.run_sequence import run_sequence

    marker = tmp_path / "stage2"
    plan = {"stages": [
        dict(name="first", command=[sys.executable, "-c", "raise SystemExit(7)"], timeout_s=5),
        dict(name="second", command=[sys.executable, "-c",
             f"from pathlib import Path; Path({str(marker)!r}).touch()"], timeout_s=5),
    ]}
    events = {}
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_sequence(plan, lambda name, event: events.update({name: json.loads(json.dumps(event))}))
    assert failure.value.returncode == 7
    assert set(events) == {"00_started", "00_failed"}
    assert events["00_failed"]["state"] == "failed"
    assert not marker.exists()


def test_run_sequence_timeout_stops_process_group(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    from infra.anyscale.run_sequence import run_stage

    pid_file = tmp_path / "pid"
    terminated = tmp_path / "terminated"
    script = ("import os, signal, time\nfrom pathlib import Path\n"
              f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
              "def stop(signum, frame):\n"
              f"    Path({str(terminated)!r}).touch()\n"
              "    raise SystemExit(0)\n"
              "signal.signal(signal.SIGTERM, stop)\ntime.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_stage(dict(name="timeout", command=[sys.executable, "-c", script], timeout_s=1))
    assert time.monotonic() - started < 10
    assert terminated.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), signal.SIGCONT)


def test_run_sequence_success_passes_environment(tmp_path):
    import sys
    from infra.anyscale.run_sequence import run_sequence

    output = tmp_path / "value"
    script = ("import os\nfrom pathlib import Path\n"
              f"Path({str(output)!r}).write_text(os.environ['PLASMID_SEQUENCE_TEST'])\n")
    events = []
    run_sequence({"stages": [dict(name="success", command=[sys.executable, "-c", script],
                  env={"PLASMID_SEQUENCE_TEST": "passed"}, timeout_s=5)]},
                 lambda name, event: events.append(name))
    assert output.read_text() == "passed"
    assert events == ["00_started", "00_succeeded"]


@pytest.mark.parametrize("require_export", [False, True])
def test_evaluation_rejects_wrong_checkpoint_step(tmp_path, require_export):
    import json
    from infra.anyscale.run_evaluation import verify_checkpoint
    from plasmid_llm.provenance import file_sha256

    names = ("model.safetensors", "config.json", "vocab.json", "tokenizer_config.json")
    for name in names:
        (tmp_path / name).write_bytes(b"checkpoint contents")
    if require_export:
        (tmp_path / "export_manifest.json").write_text(json.dumps(dict(
            training=dict(global_step=1000, requested_steps=1000, selection="last_step"),
            files={name: dict(sha256=file_sha256(tmp_path / name)) for name in names})))
    else:
        torch.save({"step": 1000}, tmp_path / "training_state.pt")
    verify_checkpoint(tmp_path, 1000, require_export)
    with pytest.raises(ValueError, match="step"):
        verify_checkpoint(tmp_path, 2000, require_export)
    if require_export:
        (tmp_path / "model.safetensors").write_bytes(b"modified weights")
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_checkpoint(tmp_path, 1000, require_export)


def test_evaluation_archive_rejects_corruption_before_manifest(tmp_path):
    import io
    from botocore.response import StreamingBody
    from infra.anyscale.run_evaluation import archive_results

    (tmp_path / "generations.parquet").write_bytes(b"results")

    class Client:
        def __init__(self, body_bytes: bytes):
            self.keys = []
            self.body_bytes = body_bytes

        def list_objects_v2(self, **kwargs):
            return {"KeyCount": 0}

        def put_object(self, **kwargs):
            self.keys.append(kwargs["Key"])

        def get_object(self, **kwargs):
            return {"Body": StreamingBody(io.BytesIO(self.body_bytes), len(self.body_bytes))}

    bad_client = Client(b"corrupted")
    with pytest.raises(ValueError, match="readback failed"):
        archive_results(bad_client, tmp_path, "s3://bucket/evaluation")
    assert "evaluation/evaluation_manifest.json" not in bad_client.keys

    good_client = Client(b"results")
    manifest = archive_results(good_client, tmp_path, "s3://bucket/evaluation")
    assert "generations.parquet" in manifest["files"]
    assert "evaluation/evaluation_manifest.json" in good_client.keys


def test_sharded_verify_split_passes_clean_and_catches_rotated_leak(tmp_path):
    import json
    from pathlib import Path
    from pretraining.build_cluster_split import canonical_circle, sha256_file
    from pretraining.verify_cluster_split import verify_split

    # Three mutually unrelated 150 bp sequences so megablast E-value << 1e-10 on self-hits.
    block = "ATGCGTACCTAGCTGATCGATCGTAGCTAGCTAGCATCGATCGTAGCTAGCTA"
    seq_a = (block + "CGATACGATACGATACGATCGATCGATCGATGCTAGCTAGCATCGAT" + "TTAGGCCAATGCAATGCAATGCAATGCAATGCAATGCAATGCA")[:150]
    seq_b = ("GGGAAATTTCCCAAAGGGTTTCCCGGGAAATTTCCCGGGAAATTTCC" + "TAGCTAGCTAGCTAGCTAGCTGATCGATCGATCGATCGATCGATCG" + "CCAATTGGCCAATTGGCCAATTGGCCAATTGGCCAATTGGCCAATTGGCCAAT")[:150]
    seq_c = ("ACGTACGTACGTACGTAGCTAGCTAGCTAGCTGATCGATCGATCGAT" + "TGCATGCATGCATGCATGCATGCAACTAGCTGACTGACTGACTGAC" + "GTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")[:150]

    def write_split(split_dir: Path, test_seq: str) -> None:
        split_dir.mkdir(parents=True)
        rows = {
            "train": [("tr_0", "p_tr", "c_tr", seq_a)],
            "validation": [("val_0", "p_val", "c_val", seq_b)],
            "test": [("te_0", "p_te", "c_te", test_seq)],
        }
        parts = {}
        for part, entries in rows.items():
            df = pd.DataFrame(
                entries, columns=["sequence_id", "plasmid_id", "cluster_id", "sequence"]
            )
            p = split_dir / f"{part}.parquet"
            df.to_parquet(p, index=False)
            parts[part] = {"sha256": sha256_file(p)}
        manifest = {"identity": 0.9, "coverage": 0.9, "partitions": parts}
        (split_dir / "manifest.json").write_text(json.dumps(manifest))

    clean_dir = tmp_path / "clean"
    write_split(clean_dir, seq_c)
    report = verify_split(clean_dir, tmp_path / "clean_audit", threads=2)
    assert report["passed"] is True
    assert not any(report["findings"].values())

    leaked_seq = seq_a[55:] + seq_a[:55]
    assert canonical_circle(leaked_seq) == canonical_circle(seq_a)
    leaked_dir = tmp_path / "leaked"
    write_split(leaked_dir, leaked_seq)
    with pytest.raises(ValueError, match="Cross-partition homology"):
        verify_split(leaked_dir, tmp_path / "leaked_audit", threads=2)


def test_refine_cluster_split_resolves_multi_hsp_insert_leakage(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from pretraining.build_cluster_split import sha256_file
    from pretraining.refine_cluster_split import refine_split
    from pretraining.verify_cluster_split import verify_split

    def make_dna(tag: str, length: int) -> str:
        bases = []
        counter = 0
        while len(bases) < length:
            digest = hashlib.sha256(f"{tag}:{counter}".encode()).digest()
            for b in digest:
                bases.append("ACGT"[b & 3])
                if len(bases) == length:
                    break
            counter += 1
        return "".join(bases)

    # Two 90 bp arms separated by a 15 bp unrelated insert -> 180/195 = 92.3% union coverage.
    arm_left = make_dna("arm_left", 90)
    arm_right = make_dna("arm_right", 90)
    seq_val_leak = arm_left + make_dna("insert_a", 15) + arm_right
    seq_tr_leak = arm_left + make_dna("insert_b", 15) + arm_right

    base_dir = tmp_path / "base_split"
    base_dir.mkdir(parents=True)
    # Deterministic partition hashes under seed=42, val_fraction=0.25, test_fraction=0.25:
    # seq_0, seq_2 -> validation; seq_1, seq_3 -> train; seq_4 -> test.
    rows_by_part = {
        "train": [
            ("seq_1", "p_tr1", "seq_1", make_dna("tr_anchor", 195)),
            ("seq_3", "p_tr2", "seq_3", seq_tr_leak),
        ],
        "validation": [
            ("seq_0", "p_val1", "seq_0", make_dna("val_anchor", 195)),
            ("seq_2", "p_val2", "seq_2", seq_val_leak),
        ],
        "test": [
            ("seq_4", "p_te1", "seq_4", make_dna("te_anchor", 195)),
        ],
    }
    part_manifest = {}
    for part, entries in rows_by_part.items():
        df = pd.DataFrame(entries, columns=["sequence_id", "plasmid_id", "cluster_id", "sequence"])
        p = base_dir / f"{part}.parquet"
        df.to_parquet(p, index=False)
        part_manifest[part] = {"sha256": sha256_file(p)}

    manifest = {
        "seed": 42,
        "identity": 0.9,
        "coverage": 0.9,
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "partitions": part_manifest,
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Cross-partition homology"):
        verify_split(base_dir, base_dir / "audit", threads=2)
    initial_report = json.loads((base_dir / "audit" / "report.json").read_text())
    assert len(initial_report["findings"]["validation_vs_train"]) == 1

    refined_dir = tmp_path / "refined_split"
    refined_manifest = refine_split(base_dir, refined_dir, threads=2)
    assert refined_manifest["refinement_iterations"] >= 1

    # Merged pair (seq_2, seq_3) must share one cluster_id and partition.
    all_parts = []
    for part in ("train", "validation", "test"):
        df = pd.read_parquet(refined_dir / f"{part}.parquet")
        df["part"] = part
        all_parts.append(df)
    merged = pd.concat(all_parts, ignore_index=True).set_index("sequence_id")
    assert merged.loc["seq_2", "cluster_id"] == merged.loc["seq_3", "cluster_id"]
    assert merged.loc["seq_2", "part"] == merged.loc["seq_3", "part"]

    clean_report = verify_split(refined_dir, refined_dir / "audit", threads=2)
    assert clean_report["passed"] is True
    assert not any(clean_report["findings"].values())


def test_cluster_export_readback_uses_real_streaming_body(tmp_path):
    import io
    import json
    from botocore.response import StreamingBody
    from infra.anyscale.run_cluster_pretraining import export_final_model

    final = tmp_path / "final"
    final.mkdir()
    payloads = {
        "model.safetensors": b"weights",
        "config.json": b"{}",
        "vocab.json": b"{}",
        "tokenizer_config.json": b"{}",
        "training_report.json": json.dumps(
            dict(selection="last_step", global_step=100, requested_steps=100)
        ).encode("utf-8"),
    }
    for name, data in payloads.items():
        (final / name).write_bytes(data)

    class Client:
        def __init__(self, corrupt: bool = False):
            self.stored = {}
            self.corrupt = corrupt

        def put_object(self, **kwargs):
            body = kwargs["Body"]
            self.stored[kwargs["Key"]] = body.read() if hasattr(body, "read") else bytes(body)

        def get_object(self, **kwargs):
            data = b"tampered" if self.corrupt else self.stored[kwargs["Key"]]
            return {"Body": StreamingBody(io.BytesIO(data), len(data))}

    ok = Client()
    manifest = export_final_model(ok, final, "s3://bucket/export/final")
    assert set(manifest["files"]) == set(payloads)
    assert "export/final/export_manifest.json" in ok.stored
    assert (tmp_path / "export_manifest.json").exists()

    (tmp_path / "export_manifest.json").unlink()
    bad = Client(corrupt=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_final_model(bad, final, "s3://bucket/export/final")
    assert "export/final/export_manifest.json" not in bad.stored


def test_wait_for_base_export_handles_ready_failed_and_timeout():
    import io
    import json
    from botocore.exceptions import ClientError
    from botocore.response import StreamingBody
    from infra.anyscale.wait_for_base_export import REQUIRED_EXPORT_FILES, wait_for_export

    valid_manifest = {
        "training": {"selection": "last_step", "global_step": 100000},
        "files": {name: {"sha256": "a" * 64, "bytes": 12} for name in REQUIRED_EXPORT_FILES},
    }

    class Client:
        def __init__(self, has_manifest: bool, has_failure: bool):
            self.has_manifest = has_manifest
            self.has_failure = has_failure

        def _not_found(self, op: str):
            return ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, op)

        def get_object(self, **kwargs):
            if not self.has_manifest:
                raise self._not_found("GetObject")
            raw = json.dumps(valid_manifest).encode("utf-8")
            return {"Body": StreamingBody(io.BytesIO(raw), len(raw))}

        def head_object(self, **kwargs):
            if not self.has_failure:
                raise self._not_found("HeadObject")
            return {}

    clock = [0.0]

    def tick() -> float:
        return clock[0]

    def advance(dt: float) -> None:
        clock[0] += dt

    ready = wait_for_export(
        Client(has_manifest=True, has_failure=False),
        "s3://b/export",
        "s3://b/events",
        monotonic_fn=tick,
        sleep_fn=advance,
    )
    assert ready["training"]["global_step"] == 100000

    with pytest.raises(RuntimeError, match="aborting"):
        wait_for_export(
            Client(has_manifest=False, has_failure=True),
            "s3://b/export",
            "s3://b/events",
            monotonic_fn=tick,
            sleep_fn=advance,
        )

    clock[0] = 0.0
    with pytest.raises(TimeoutError, match="Timed out"):
        wait_for_export(
            Client(has_manifest=False, has_failure=False),
            "s3://b/export",
            "s3://b/events",
            poll_seconds=5.0,
            max_wait_seconds=10.0,
            monotonic_fn=tick,
            sleep_fn=advance,
        )


def test_circular_protein_filter_keeps_boundary_orfs_once(tmp_path):
    from Bio import SeqIO
    from eval.scripts.annotate import select_circular_proteins

    proteins = tmp_path / "doubled.faa"
    proteins.write_text(
        ">gen_0_1 # 80 # 120 # 1 # partial=00\nMAAA*\n"
        ">gen_0_2 # 180 # 200 # 1 # partial=10\nMAA\n"
        ">gen_0_3 # 30 # 60 # -1 # partial=00\nMCCC*\n"
        ">gen_0_4 # 130 # 160 # -1 # partial=00\nMCCC*\n"
        ">gen_0_5 # 10 # 150 # 1 # partial=00\nMDDD*\n"
        ">gen_0_6 # 80 # 120 # 1 # partial=00\nMAAA*\n")
    output = tmp_path / "retained.faa"
    counts = select_circular_proteins(proteins, {"gen_0": 100}, output)
    assert [r.id for r in SeqIO.parse(output, "fasta")] == ["gen_0_1", "gen_0_3"]
    assert counts == dict(input_orfs=6, retained_orfs=2,
                          duplicate_or_outside_first_copy=3, longer_than_circle=1)


def test_circular_prodigal_records_short_sequences_without_calling_binary(tmp_path):
    import json
    from eval.scripts.annotate import run_prodigal

    fasta = tmp_path / "empty.fasta"
    fasta.write_text(">gen_0\n\n>gen_1\nACGT\n")
    gff = run_prodigal(str(fasta), str(tmp_path / "results"), circular=True)
    assert gff.name == "prodigal_doubled.gff"
    assert (gff.parent / "prodigal_proteins.faa").read_text() == ""
    counts = json.loads((gff.parent / "circular_filter.json").read_text())
    assert counts["input_sequences"] == counts["shorter_than_20bp"] == 2
    assert counts["retained_orfs"] == 0


def test_annotation_failure_is_not_scored_as_zero(tmp_path):
    import json
    from experiments.neurips_rebuttal.common import load_annotations

    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([dict(seq_id="gen_0", features=[], error="annotation process failed")]))
    with pytest.raises(ValueError, match="cannot be scored"):
        load_annotations(path)


def test_training_seed_summary_validates_pairing_and_averages_decodes():
    from experiments.neurips_rebuttal.common import summarize_paired_training_seeds

    panel = pd.DataFrame([dict(sample_id=f"p{i}", cluster_id=f"c{i}", prompt=f"prompt{i}")
                          for i in range(3)])
    rows = [dict(row, training_seed=training_seed, seed=decode_seed,
                 score=(decode_seed - 42) / 10)
            for row in panel.to_dict("records")
            for training_seed in (42, 43, 44) for decode_seed in (42, 43, 44, 45)]
    baseline = pd.DataFrame(rows)
    treatment = baseline.copy()
    treatment["score"] += .25
    result = summarize_paired_training_seeds(
        baseline.sample(frac=1, random_state=8), treatment, panel, "score", [42, 43, 44, 45], iters=100)
    assert result["delta"] == pytest.approx(.25)
    assert result["baseline_mean"] == pytest.approx(.15)
    assert result["paired_cluster_training_seed_ci95"] == pytest.approx([.25, .25])
    assert result["n_prompts"] == result["n_clusters"] == 3
    with pytest.raises(ValueError, match="frozen panel"):
        summarize_paired_training_seeds(baseline.iloc[1:], treatment, panel, "score", [42, 43, 44, 45])
    wrong = treatment.copy()
    wrong.loc[0, "cluster_id"] = "wrong-cluster"
    with pytest.raises(ValueError, match="frozen panel"):
        summarize_paired_training_seeds(baseline, wrong, panel, "score", [42, 43, 44, 45])
    fixed = baseline[baseline.training_seed == 42]
    fixed_result = summarize_paired_training_seeds(
        fixed, treatment, panel, "score", [42, 43, 44, 45], fixed_baseline=True, iters=100)
    assert fixed_result["baseline_training_seeds"] == [42]
    assert fixed_result["delta"] == pytest.approx(.25)


def test_static_cache_generate_simple_matches_eager_with_left_padding_and_eos():
    from plasmid_llm.models.hf_plasmid_lm.configuration_plasmid_lm import PlasmidLMConfig
    from plasmid_llm.models.hf_plasmid_lm.modeling_plasmid_lm import PlasmidLMForCausalLM

    torch.manual_seed(7)
    cfg = PlasmidLMConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = PlasmidLMForCausalLM(cfg).eval()
    # Bias token 2 (EOS) upward so EOS triggers early across rows.
    with torch.no_grad():
        model.lm_head.weight[2].add_(1.5)
    input_ids = torch.tensor([
        [0, 0, 1, 5, 6, 7],
        [1, 8, 9, 10, 11, 12],
    ])
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])

    torch.manual_seed(123)
    eager_out = model.generate_simple(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=16,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        eos_token_id=2,
        pad_token_id=0,
        use_static_cache=False,
    )
    torch.manual_seed(123)
    static_out = model.generate_simple(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=16,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        eos_token_id=2,
        pad_token_id=0,
        use_static_cache=True,
    )
    torch.testing.assert_close(static_out, eager_out, rtol=0, atol=0)
    assert (static_out[:, 6:] == 2).any()

    # Check generate_compiled alias preserves identical sampling behavior too.
    torch.manual_seed(123)
    compiled_out = model.generate_compiled(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=16,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.testing.assert_close(compiled_out, eager_out, rtol=0, atol=0)


def test_right_padded_causal_matches_masked_forward_and_gradients():
    from plasmid_llm.models.hf_plasmid_lm.configuration_plasmid_lm import PlasmidLMConfig
    from plasmid_llm.models.hf_plasmid_lm.modeling_plasmid_lm import PlasmidLMForCausalLM

    torch.manual_seed(11)
    cfg = PlasmidLMConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model_masked = PlasmidLMForCausalLM(cfg).train()
    model_causal = PlasmidLMForCausalLM(cfg).train()
    model_causal.load_state_dict(model_masked.state_dict())
    model_causal.model.right_padded_causal = True

    input_ids = torch.tensor([
        [1, 10, 11, 12, 13, 14, 2, 0, 0, 0],
        [1, 20, 21, 22, 23, 24, 25, 26, 2, 0],
        [1, 30, 31, 32, 2, 0, 0, 0, 0, 0],
    ])
    attention_mask = (input_ids != 0).long()
    labels = input_ids.clone()
    labels[:, :2] = -100  # mask prompt positions
    labels[attention_mask == 0] = -100

    out_masked = model_masked(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out_causal = model_causal(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    torch.testing.assert_close(out_causal.loss, out_masked.loss, rtol=1e-5, atol=1e-6)

    out_masked.loss.backward()
    out_causal.loss.backward()
    for (name, pm), (_, pc) in zip(model_masked.named_parameters(), model_causal.named_parameters()):
        if pm.grad is None and pc.grad is None:
            continue
        assert pm.grad is not None and pc.grad is not None, name
        torch.testing.assert_close(pc.grad, pm.grad, rtol=1e-5, atol=1e-6, msg=name)


def test_fast_kmer_encode_and_complete_orfs_exact_equivalence(tmp_path):
    import json
    import random
    import re
    from Bio.Seq import Seq
    from plasmid_llm.models.hf_plasmid_lm.tokenization_kmer import PlasmidKmerTokenizer, build_kmer_vocab
    from post_training.scorers.plannotate_v2 import PlannotateV2Scorer, STARTS, STOPS, complete_orfs

    vocab = build_kmer_vocab(["<BOS>", "<EOS>", "<UNK>", "<PAD>", "<SEP>", "<AMR_AMPICILLIN>", "<SEQ>"], k=6)
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    tok = PlasmidKmerTokenizer(str(vocab_path), k=6, stride=3)

    samples = [
        "<BOS> <AMR_AMPICILLIN> <SEQ>",
        "ATGGCTAGCTAGCATCGATCGTAGCTAGCTA",
        "  atg cnnn gct agc nnn tag ",
        "ACGT",
        "<BOS> <AMR_AMPICILLIN> <SEP> ATGNNNATGCGTACG <EOS>",
    ]
    for s in samples:
        assert tok.encode_fast(s) == tok.encode(s)

    def ref_complete_orfs(dna: str, min_aa: int = 30) -> list[str]:
        proteins: set[str] = set()
        for strand in (dna, str(Seq(dna).reverse_complement())):
            doubled = strand + strand
            for frame in range(3):
                starts: list[int] = []
                for pos in range(frame, len(doubled) - 2, 3):
                    codon = doubled[pos:pos + 3]
                    if codon in STOPS or re.search("[^ACGT]", codon):
                        if codon in STOPS:
                            for start in starts:
                                length = pos - start
                                if min_aa * 3 <= length and length + 3 <= len(dna):
                                    peptide = str(Seq(doubled[start:pos]).translate(table=11))
                                    proteins.add("M" + peptide[1:])
                        starts = []
                    elif codon in STARTS and pos < len(dna):
                        starts.append(pos)
        return sorted(proteins)

    rng = random.Random(42)
    dna_cases = [
        "".join(rng.choices("ACGT", k=1200)),
        # Construct ORF wrapping across origin + nested GTG start + N in another frame
        "GCTAGCTA" * 15 + "TAANNN" + "ATG" + "GCTGCT" * 20 + "GTG" + "GCT" * 35,
        "atg" + "gct" * 40 + "taa",
    ]
    for seq in dna_cases:
        assert complete_orfs(seq.upper()) == ref_complete_orfs(seq.upper())

    # Deduplicated parallel scorer must match scoring duplicates identically.
    gene_dna = "ATG" + "".join(rng.choices(["GCT", "GAA", "AAG", "CTG", "ACC"], k=80)) + "TAA"
    gene_prot = str(Seq(gene_dna).translate(table=11, cds=True))
    ori_dna = "".join(rng.choices("ACGT", k=240))
    registry = pd.DataFrame([
        dict(token="<AMR_AMPICILLIN>", sseqid="blaTEM-1", seq_type="dna", sequence=gene_dna),
        dict(token="<AMR_AMPICILLIN>", sseqid="blaTEM-1_prot", seq_type="protein", sequence=gene_prot),
        dict(token="<ORI_COLE1>", sseqid="ColE1", seq_type="nucleotide", sequence=ori_dna),
    ])
    scorer = PlannotateV2Scorer(motif_registry_df=registry, db_dir=str(tmp_path), num_threads=2)
    prompts = ["<BOS> <AMR_AMPICILLIN> <ORI_COLE1> <SEQ>"] * 4
    cands = [
        gene_dna + ori_dna,
        gene_dna + ori_dna,  # exact duplicate
        "ACGT",              # shorter than 20 bp
        ori_dna + "TAGCATCGATCGATCGATCGATCGATCGAT",
    ]
    detailed = scorer.score_batch_detailed(prompts, cands)
    assert detailed[0] == detailed[1]
    assert detailed[0]["intact_orf_recall"] == 1.0
    assert detailed[2]["reward"] == 0.0
    assert detailed[3] == scorer.score_sequence_detailed(prompts[3], cands[3])


def test_orthogonal_supports_amrfinder_4_columns_and_methods():
    from experiments.neurips_rebuttal.orthogonal import is_intact_hit

    complete = {"gen_0_1"}
    for method in ("EXACTP", "BLASTP", "ALLELEP"):
        row_v4 = {"Protein id": "gen_0_1", "Method": method, "% Coverage of reference": "95.2"}
        assert is_intact_hit(row_v4, complete)
    assert not is_intact_hit(
        {"Protein id": "gen_0_1", "Method": "PARTIALP", "% Coverage of reference": "95.2"}, complete
    )
    assert is_intact_hit(
        {"Protein id": "gen_0_1", "Method": "BLAST", "% Coverage of reference protein": "92.0"}, complete
    )
    with pytest.raises(ValueError, match="missing columns"):
        is_intact_hit({"Protein id": "gen_0_1", "Method": "BLASTP"}, complete)


def test_summarize_rigor_evaluation_end_to_end_and_helpers(tmp_path):
    from experiments.neurips_rebuttal.summarize_rigor_evaluation import (
        classify_copy_low,
        detect_prodigal_peptide_tags,
        holm_adjust,
        parse_dustmasker_intervals,
        score_single_panel_dataframe,
        summarize_rigor_evaluation,
    )

    intervals = tmp_path / "dust.intervals"
    intervals.write_text(">gen_0\n0 - 9\n5 - 19\n>gen_1\n")
    fracs = parse_dustmasker_intervals(intervals, {"gen_0": 100, "gen_1": 50, "gen_2": 0})
    assert fracs == pytest.approx({"gen_0": 0.2, "gen_1": 0.0, "gen_2": 0.0})

    faa = tmp_path / "proteins.faa"
    faa.write_text(
        ">gen_0_1 # 1 # 90 # 1 # ID=1;partial=00;start_type=ATG\nMAAAADYKDDDDK*\n"
        ">gen_0_2 # 1 # 60 # 1 # ID=2;partial=10;start_type=Edge\nMHHHHHH*\n"
    )
    assert detect_prodigal_peptide_tags(faa) == {"gen_0": {"<TAG_FLAG>"}}

    assert classify_copy_low({"<ORI_P15A>"})["copy_low_compatible_only"] == 1.0
    conflict = classify_copy_low({"<ORI_P15A>", "<ORI_COLE1>"})
    assert conflict["copy_low_conflicting_high_and_low"] == 1.0
    assert conflict["copy_low_compatible_only"] == 0.0
    assert conflict["multi_bacterial_origin"] == 1.0

    assert holm_adjust([0.01, 0.04, 0.03, 0.002]) == pytest.approx([0.03, 0.06, 0.06, 0.008])

    panel_df = pd.DataFrame([
        dict(sample_id="s0", cluster_id="c0", prompt="<BOS><COPY_LOW><ORI_P15A><AMR_AMPICILLIN><TAG_FLAG><SEQ>",
             panel_main=True, minority_panels=["<ORI_P15A>", "<COPY_LOW>"]),
        dict(sample_id="s1", cluster_id="c1", prompt="<BOS><ORI_PSC101><AMR_CHLORAMPHENICOL><TAG_HIS><SEQ>",
             panel_main=True, minority_panels=["<ORI_PSC101>", "<AMR_CHLORAMPHENICOL>"]),
        dict(sample_id="s2", cluster_id="c2", prompt="<BOS><ORI_COLE1><AMR_AMPICILLIN><SEQ>",
             panel_main=False, minority_panels=[]),
    ])
    manifest = {"rows": 3, "main_clusters": 2}

    def make_panel_metrics(model_name, family, tseed, dseed, *, good=True):
        gen_df = pd.DataFrame([
            dict(sample_id=r.sample_id, cluster_id=r.cluster_id, prompt=r.prompt,
                 sequence="ACGT" * 250, length=1000, gc=0.5, has_eos=True, truncated=False)
            for r in panel_df.itertuples()
        ])
        feats_good = [
            {"Feature": "p15A ori", "Type": "rep_origin", "percmatch": 99.0, "fragment": False},
            {"Feature": "AmpR", "Type": "CDS", "percmatch": 99.0, "fragment": False},
            {"Feature": "T7 promoter", "Type": "promoter", "percmatch": 99.0, "fragment": False},
        ]
        feats_ok = [
            {"Feature": "pSC101 ori", "Type": "rep_origin", "percmatch": 99.0, "fragment": False},
            {"Feature": "CmR", "Type": "CDS", "percmatch": 99.0, "fragment": False},
        ]
        ann = {
            "gen_0": feats_good if good else [feats_good[1]],
            "gen_1": feats_ok if good else [],
            "gen_2": feats_good,
        }
        orth_all = pd.DataFrame([
            dict(sequence_id="gen_0", prompt=panel_df.iloc[0]["prompt"], requested=["<AMR_AMPICILLIN>"],
                 detected=["<AMR_AMPICILLIN>"] if good else [], recall=1.0 if good else 0.0, all_requested=good, intact_only=False),
            dict(sequence_id="gen_1", prompt=panel_df.iloc[1]["prompt"], requested=["<AMR_CHLORAMPHENICOL>"],
                 detected=["<AMR_CHLORAMPHENICOL>"] if good else [], recall=1.0 if good else 0.0, all_requested=good, intact_only=False),
            dict(sequence_id="gen_2", prompt=panel_df.iloc[2]["prompt"], requested=["<AMR_AMPICILLIN>"],
                 detected=["<AMR_AMPICILLIN>"], recall=1.0, all_requested=True, intact_only=False),
        ])
        orth_intact = orth_all.assign(intact_only=True)
        return score_single_panel_dataframe(
            panel_df=panel_df,
            generations_df=gen_df,
            plannotate_annotations=ann,
            orthogonal_all_df=orth_all,
            orthogonal_intact_df=orth_intact,
            peptide_tags={"gen_0": {"<TAG_FLAG>"} if good else set(), "gen_1": {"<TAG_HIS>"} if good else set()},
            low_complexity={"gen_0": 0.05, "gen_1": 0.02, "gen_2": 0.0},
            model_name=model_name,
            model_family=family,
            training_seed=tseed,
            decode_seed=dseed,
        )

    ref_df = make_panel_metrics("reference_control", "reference_control", 0, 0, good=True)
    frames = []
    for dseed in (42, 43, 44, 45):
        frames.append(make_panel_metrics("hf_base", "hf_base", 42, dseed, good=False))
        frames.append(make_panel_metrics("cluster_base_s42", "cluster_base_s42", 42, dseed, good=False))
        for tseed in (42, 43, 44):
            frames.append(make_panel_metrics(f"exp1_v1_s{tseed}", "exp1_v1", tseed, dseed, good=False))
            frames.append(make_panel_metrics(f"exp1_v2_s{tseed}", "exp1_v2", tseed, dseed, good=True))
            frames.append(make_panel_metrics(f"exp2_v2_s{tseed}", "exp2_v2", tseed, dseed, good=True))

    out_dir = tmp_path / "results"
    summary = summarize_rigor_evaluation(
        panel_df=panel_df,
        panel_manifest=manifest,
        ref_df=ref_df,
        model_frames=frames,
        decode_seeds=[42, 43, 44, 45],
        iters=50,
        output_dir=out_dir,
    )
    comp = summary["paired_comparisons"]["exp1_v2_vs_exp1_v1:main"]
    assert comp["useful_05"]["delta"] == pytest.approx(1.0)
    assert comp["useful_05"]["holm_p_value"] is not None
    assert comp["tag_recall_prodigal_peptide"]["delta"] == pytest.approx(1.0)
    assert (out_dir / "REPORT.md").is_file()
    assert (out_dir / "tables/paired_comparisons.csv").is_file()


def test_cpu_shard_manifest_wait_and_safe_tar_extract(tmp_path):
    import io
    import tarfile
    from botocore.exceptions import ClientError
    from botocore.response import StreamingBody
    from infra.anyscale.run_cpu_annotation_shard import (
        safe_extract_tar,
        wait_for_peer_annotations,
        wait_for_s3_manifest,
    )

    class Client:
        def __init__(self, objects):
            self.objects = objects

        def _nf(self, op):
            return ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, op)

        def get_object(self, **kwargs):
            key = kwargs["Key"]
            if key not in self.objects:
                raise self._nf("GetObject")
            raw = json_bytes(self.objects[key])
            return {"Body": StreamingBody(io.BytesIO(raw), len(raw))}

        def head_object(self, **kwargs):
            if kwargs["Key"] not in self.objects:
                raise self._nf("HeadObject")
            return {}

    import json as _json

    def json_bytes(obj):
        return _json.dumps(obj).encode("utf-8")

    ok_manifest = {
        "files": {
            "generations.parquet": {"sha256": "a" * 64},
            "generations.fasta": {"sha256": "b" * 64},
            "generation_summary.json": {"sha256": "c" * 64},
            "config.json": {"sha256": "d" * 64},
        }
    }
    ready = wait_for_s3_manifest(
        Client({"eval/decode_s42/evaluation_manifest.json": ok_manifest}),
        "s3://b/eval/decode_s42/evaluation_manifest.json",
    )
    assert "generations.parquet" in ready["files"]

    with pytest.raises(RuntimeError, match="aborting CPU wait"):
        wait_for_s3_manifest(
            Client({"wf/01_failed.json": {"state": "failed"}}),
            "s3://b/eval/decode_s42/evaluation_manifest.json",
            workflow_events_s3_uri="s3://b/wf",
            stage_index=1,
        )

    with pytest.raises(RuntimeError, match="Peer CPU shard"):
        wait_for_peer_annotations(
            Client({"ann/_shards/eval_cpu_shard1_v1.json": {"state": "failed", "error": "boom"}}),
            "s3://b/ann",
            ["reference_control/annotation_manifest.json"],
            ["eval_cpu_shard1_v1"],
        )

    # Safe tar extract rejects path traversal and strips prefix cleanly.
    bad_tar = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe tar entry path"):
        safe_extract_tar(bad_tar, tmp_path / "out_bad")

    good_tar = tmp_path / "good.tar.gz"
    with tarfile.open(good_tar, "w:gz") as tf:
        info = tarfile.TarInfo("plannotate/data/BLAST_dbs/db.txt")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"abc"))
    extracted = safe_extract_tar(
        good_tar, tmp_path / "out_good",
        strip_prefix="plannotate/", only_subpath="plannotate/data/BLAST_dbs/",
    )
    assert (tmp_path / "out_good/data/BLAST_dbs/db.txt").read_bytes() == b"abc"
    assert len(extracted) == 1


def test_cpu_ray_finish_discovery_ordering_and_plan():
    import hashlib
    import io
    import json as _json
    from pathlib import Path
    from botocore.exceptions import ClientError
    from botocore.response import StreamingBody
    from infra.anyscale.run_cpu_annotation_shard import parse_s3_uri
    from infra.anyscale.run_cpu_ray_finish import (
        discover_reusable_panels,
        order_tasks_lpt,
    )

    class BytesClient:
        def __init__(self, objects):
            self.objects = objects

        def get_object(self, **kwargs):
            key = kwargs["Key"]
            if key not in self.objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            raw = self.objects[key]
            return {"Body": StreamingBody(io.BytesIO(raw), len(raw))}

    good_parquet = b"parquet-payload"
    good_sha = hashlib.sha256(good_parquet).hexdigest()
    good_manifest = _json.dumps({
        "files": {"per_sequence_metrics.parquet": {"sha256": good_sha, "bytes": len(good_parquet)}}
    }).encode("utf-8")

    tasks = [
        {"relpath": "hf_base/decode_s42", "model_family": "hf_base", "training_seed": 42, "decode_seed": 42},
        {"relpath": "exp1_v1_s42/decode_s42", "model_family": "exp1_v1", "training_seed": 42, "decode_seed": 42},
        {"relpath": "exp1_v2_s42/decode_s42", "model_family": "exp1_v2", "training_seed": 42, "decode_seed": 42},
        {"relpath": "cluster_base_s42/decode_s42", "model_family": "cluster_base_s42", "training_seed": 42, "decode_seed": 42},
        {"relpath": "exp2_v2_s42/decode_s42", "model_family": "exp2_v2", "training_seed": 42, "decode_seed": 42},
    ]

    # Reuses verified reference_control, ignores partial unmanifested upload on exp1_v1, leaves rest pending.
    client_partial = BytesClient({
        "ann_v1/reference_control/annotation_manifest.json": good_manifest,
        "ann_v1/reference_control/per_sequence_metrics.parquet": good_parquet,
        "ann_v1/exp1_v1_s42/decode_s42/plannotate_results.json": b"{}",
    })
    reused, pending = discover_reusable_panels(
        client_partial, ["s3://b/ann_v1"], tasks, annotate_reference_control=True
    )
    assert set(reused) == {"reference_control"}
    assert reused["reference_control"]["source"] == "reused"
    assert [t["relpath"] for t in pending] == [t["relpath"] for t in tasks]

    # Corrupt checksum on a published manifest raises ValueError.
    corrupt_manifest = _json.dumps({
        "files": {"per_sequence_metrics.parquet": {"sha256": "0" * 64, "bytes": len(good_parquet)}}
    }).encode("utf-8")
    client_corrupt = BytesClient({
        "ann_v1/hf_base/decode_s42/annotation_manifest.json": corrupt_manifest,
        "ann_v1/hf_base/decode_s42/per_sequence_metrics.parquet": good_parquet,
    })
    with pytest.raises(ValueError, match="Checksum mismatch"):
        discover_reusable_panels(client_corrupt, ["s3://b/ann_v1"], tasks, annotate_reference_control=False)

    # LPT ordering places reference_control, then exp2_v2 -> cluster_base_s42 -> exp1_v2 -> exp1_v1 -> hf_base.
    ordered = order_tasks_lpt([
        {"relpath": "reference_control", "model_family": "reference_control", "training_seed": 0, "decode_seed": 0},
        *tasks,
    ])
    assert [t["model_family"] for t in ordered] == [
        "reference_control",
        "exp2_v2",
        "cluster_base_s42",
        "exp1_v2",
        "exp1_v1",
        "hf_base",
    ]

    # Verify cpu_ray_finish_v1.json plan integrity.
    plan_path = Path(__file__).resolve().parents[1] / "infra/anyscale/eval_plans/cpu_ray_finish_v1.json"
    plan = _json.loads(plan_path.read_text())
    relpaths = [t["relpath"] for t in plan["tasks"]]
    assert len(relpaths) == 44
    assert len(set(relpaths)) == 44
    expected_manifests = [f"{rp}/annotation_manifest.json" for rp in ["reference_control", *relpaths]]
    assert set(plan["all_required_annotation_manifests"]) == set(expected_manifests)
    assert len(plan["all_required_annotation_manifests"]) == 45
    assert plan["reuse_annotation_s3_roots"] != [plan["write_annotation_s3_root"]]
    for uri in [*plan["reuse_annotation_s3_roots"], plan["write_annotation_s3_root"], plan["results_s3_destination"]]:
        parse_s3_uri(uri)

