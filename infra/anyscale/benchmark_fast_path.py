"""Live L40S benchmark and numerical equivalence check for source_v9 fast paths.

Runs non-destructively on one g6e.xlarge worker + m5.2xlarge head, verifies CUDA
bfloat16 equivalence for CUDA-graphed static-KV-cache generation and
right_padded_causal FlashAttention-2, benchmarks pretraining microbatches (16, 32)
with torch.compile on cluster_v3, and runs 3 end-to-end GRPO-v2 steps.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import ray
import torch

from plasmid_llm.provenance import s3_readback_sha256


@ray.remote(num_gpus=1)
def verify_cuda_equivalence() -> dict:
    from plasmid_llm.models.hf_plasmid_lm import PlasmidLMConfig, PlasmidLMForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 is required")

    device = torch.device("cuda")
    torch.manual_seed(17)
    cfg = PlasmidLMConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = PlasmidLMForCausalLM(cfg).to(device).eval()

    input_ids = torch.tensor(
        [
            [0, 0, 1, 15, 22, 33, 44, 55],
            [1, 16, 23, 34, 45, 56, 67, 78],
        ],
        device=device,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        device=device,
    )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(99)
        eager = model.generate_simple(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            temperature=1.0,
            top_k=1,
            eos_token_id=None,
            pad_token_id=0,
            use_static_cache=False,
        )
        torch.manual_seed(99)
        graphed = model.generate_simple(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            temperature=1.0,
            top_k=1,
            eos_token_id=None,
            pad_token_id=0,
            use_static_cache=True,
        )
    gen_exact_match = bool(torch.equal(eager, graphed))

    # Right-padded causal mask equivalence in fp32 and bf16.
    rp_ids = torch.tensor(
        [
            [1] + list(range(10, 140)) + [2] + [0] * 125,
            [1] + list(range(20, 220)) + [2] + [0] * 55,
            [1] + list(range(30, 180)) + [2] + [0] * 105,
            [1] + list(range(40, 250)) + [2] + [0] * 45,
        ],
        device=device,
    )
    rp_mask = (rp_ids != 0).long()
    rp_labels = rp_ids.clone()
    rp_labels[:, :4] = -100
    rp_labels[rp_mask == 0] = -100

    model.train()
    model.model.right_padded_causal = False
    loss_fp32_masked = model(input_ids=rp_ids, attention_mask=rp_mask, labels=rp_labels).loss
    model.model.right_padded_causal = True
    loss_fp32_causal = model(input_ids=rp_ids, attention_mask=rp_mask, labels=rp_labels).loss
    fp32_diff = abs(float(loss_fp32_masked.item() - loss_fp32_causal.item()))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.model.right_padded_causal = False
        loss_bf16_masked = model(input_ids=rp_ids, attention_mask=rp_mask, labels=rp_labels).loss
        model.model.right_padded_causal = True
        loss_bf16_causal = model(input_ids=rp_ids, attention_mask=rp_mask, labels=rp_labels).loss
    bf16_diff = abs(float(loss_bf16_masked.item() - loss_bf16_causal.item()))

    if not gen_exact_match:
        raise AssertionError("CUDA graph generate_simple diverged from eager under greedy top_k=1")
    if fp32_diff >= 1e-5:
        raise AssertionError(f"fp32 right_padded_causal loss diff too large: {fp32_diff}")
    if bf16_diff >= 2e-2:
        raise AssertionError(f"bf16 right_padded_causal loss diff too large: {bf16_diff}")

    return {
        "cuda_device": torch.cuda.get_device_name(0),
        "gen_greedy_exact_match": gen_exact_match,
        "fp32_causal_loss_diff": fp32_diff,
        "bf16_causal_loss_diff": bf16_diff,
    }


@ray.remote(num_gpus=1)
def benchmark_pretraining_microbatch(split_prefix: str, microbatch: int, steps: int = 25) -> dict:
    import statistics
    import pyarrow.parquet as pq
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from plasmid_llm.data import PlasmidDataset
    from plasmid_llm.models.hf_plasmid_lm import (
        PlasmidKmerTokenizer,
        PlasmidLMConfig,
        PlasmidLMForCausalLM,
        build_kmer_vocab,
    )

    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix=f"pretrain_bench_m{microbatch}_") as tmp:
        work = Path(tmp)
        split_dir = work / "split"
        split_dir.mkdir()
        train_path = split_dir / "train.parquet"
        subprocess.run(["aws", "s3", "cp", f"{split_prefix}/train.parquet", str(train_path), "--no-progress"], check=True)
        # Sample an evenly strided 4,096-row slice across train.parquet so lengths match distribution.
        sub_path = work / "train_sub.parquet"
        tbl = pq.read_table(train_path)
        stride = max(1, tbl.num_rows // 4096)
        pq.write_table(tbl.take(list(range(0, tbl.num_rows, stride))[:4096]), sub_path)

        vocab = build_kmer_vocab(
            [line.strip() for line in Path("data/special_tokens.txt").read_text().splitlines() if line.strip()],
            k=6,
        )
        vocab_path = work / "vocab.json"
        vocab_path.write_text(json.dumps(vocab))
        tok = PlasmidKmerTokenizer(str(vocab_path), k=6, stride=3)
        ds = PlasmidDataset(
            str(sub_path), tok, max_seq_len=4096, circular_rotation=True, soft_token_dropout=0.2,
        )

        padded_vocab = ((tok.vocab_size + 7) // 8) * 8
        model = PlasmidLMForCausalLM(
            PlasmidLMConfig(
                vocab_size=padded_vocab,
                hidden_size=384,
                num_hidden_layers=10,
                num_attention_heads=8,
                intermediate_size=1536,
                max_position_embeddings=16384,
                pad_token_id=tok.pad_token_id,
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id,
                tokenizer_type="kmer",
                kmer_k=6,
                kmer_stride=3,
            )
        )
        model.model.right_padded_causal = True

        step_seconds = []

        class TimingCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                self.t0 = time.perf_counter()

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                step_seconds.append(time.perf_counter() - self.t0)

        targs = TrainingArguments(
            output_dir=str(work / "out"),
            per_device_train_batch_size=microbatch,
            gradient_accumulation_steps=64 // microbatch,
            max_steps=steps,
            warmup_steps=1000,
            learning_rate=3e-4,
            weight_decay=0.1,
            max_grad_norm=1.0,
            bf16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            torch_compile=True,
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            report_to=["none"],
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds,
            processing_class=tok,
            callbacks=[TimingCallback()],
        )
        trainer.train()
        steady = step_seconds[5:]
        mean_s = statistics.mean(steady)
        return {
            "microbatch": microbatch,
            "grad_accum": 64 // microbatch,
            "steps": len(step_seconds),
            "warmup_first_step_s": step_seconds[0],
            "steady_mean_step_s": mean_s,
            "steady_median_step_s": statistics.median(steady),
            "projected_100k_hours": mean_s * 100_000 / 3600.0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }


def main() -> None:
    import wandb
    from infra.anyscale.run_anyscale import install_blast
    from plasmid_llm.utils import load_prompts_from_parquet
    from post_training.algorithms import build_algorithm
    from post_training.common.model import ModelActor
    from post_training.scorers import build_scorer

    split_prefix = os.environ["S3_SPLIT_PREFIX"].rstrip("/")
    data_prefix = os.environ["S3_DATA_PREFIX"].rstrip("/")
    output_uri = os.environ["S3_BENCHMARK_OUTPUT"]

    parsed = urlparse(output_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid S3_BENCHMARK_OUTPUT: {output_uri}")
    client = boto3.client("s3")
    out_key = parsed.path.lstrip("/")
    if client.list_objects_v2(Bucket=parsed.netloc, Prefix=out_key, MaxKeys=1).get("KeyCount", 0):
        raise FileExistsError(f"Benchmark output already exists: {output_uri}")

    install_blast()
    wandb.init(
        project="PlasmidLLM",
        name=os.environ.get("PLASMID_RUN_ID", "rigor_20261116_fast_path_benchmark_s42"),
        tags=["plasmidrl", "benchmark", "fast-path"],
    )
    ray.init(address="auto", ignore_reinit_error=True)

    failed = True
    try:
        eq = ray.get(verify_cuda_equivalence.remote())
        print("CUDA equivalence:", json.dumps(eq), flush=True)

        m16 = ray.get(benchmark_pretraining_microbatch.remote(split_prefix, 16, 25))
        print("Pretrain m=16:", json.dumps(m16), flush=True)
        m32 = ray.get(benchmark_pretraining_microbatch.remote(split_prefix, 32, 25))
        print("Pretrain m=32:", json.dumps(m32), flush=True)

        with tempfile.TemporaryDirectory(prefix="grpo_bench_") as tmp:
            tdir = Path(tmp)
            motif_path = tdir / "motif_registry_combined.parquet"
            prompts_path = tdir / "train.parquet"
            subprocess.run(["aws", "s3", "cp", f"{data_prefix}/data/motif_registry_combined.parquet", str(motif_path), "--no-progress"], check=True)
            subprocess.run(["aws", "s3", "cp", f"{data_prefix}/data/cluster_v3/train.parquet", str(prompts_path), "--no-progress"], check=True)

            prompts = load_prompts_from_parquet(str(prompts_path), filter_hard_tokens=True)[:8]
            scorer = build_scorer(
                "plannotate_v2",
                motif_registry_path=str(motif_path),
                f1_weight=0.6,
                orf_weight=0.3,
                tag_weight=0.1,
            )
            actor = ModelActor.remote(
                "McClain/PlasmidLM-kmer6",
                model_revision="f10ec7eda876c8a0217240241d4f463fb50f998a",
                learning_rate=5e-6,
                warmup_steps=50,
                max_steps=1000,
                max_grad_norm=1.0,
                bf16=True,
                seed=42,
            )
            algo = build_algorithm(
                "grpo",
                kl_coef=0.5,
                cliprange=0.2,
                num_generations=8,
                micro_batch_size=4,
                gen_kwargs={"max_new_tokens": 4096, "temperature": 0.3, "top_p": 0.95},
            )
            grpo_steps = []
            try:
                for step_idx in range(1, 4):
                    m = algo.step([actor], prompts, scorer)
                    entry = {
                        "step": step_idx,
                        "gen_s": m["time/generation"],
                        "score_s": m["time/scoring"],
                        "train_s": m["time/train"],
                        "total_s": m["time/total"],
                        "gen_tok_per_s": m["throughput/gen_tokens_per_sec"],
                        "mean_reward": m["mean_reward"],
                    }
                    grpo_steps.append(entry)
                    print(f"GRPO step {step_idx}:", json.dumps(entry), flush=True)
            finally:
                ray.kill(actor)

        steady_grpo = grpo_steps[1:]
        mean_gen = sum(x["gen_s"] for x in steady_grpo) / len(steady_grpo)
        mean_score = sum(x["score_s"] for x in steady_grpo) / len(steady_grpo)
        mean_train = sum(x["train_s"] for x in steady_grpo) / len(steady_grpo)
        mean_total = sum(x["total_s"] for x in steady_grpo) / len(steady_grpo)

        report = {
            "equivalence": eq,
            "pretraining": {
                "microbatch16": m16,
                "microbatch32": m32,
                "best_microbatch": 32 if m32["projected_100k_hours"] <= m16["projected_100k_hours"] else 16,
                "best_projected_100k_hours": min(m16["projected_100k_hours"], m32["projected_100k_hours"]),
            },
            "grpo_v2": {
                "steps": grpo_steps,
                "steady_mean_gen_s": mean_gen,
                "steady_mean_score_s": mean_score,
                "steady_mean_train_s": mean_train,
                "steady_mean_total_s": mean_total,
                "projected_1000_steps_hours": mean_total * 1000.0 / 3600.0,
            },
        }
        payload = (json.dumps(report, indent=2) + "\n").encode()
        import hashlib
        digest = hashlib.sha256(payload).hexdigest()
        client.put_object(
            Bucket=parsed.netloc,
            Key=out_key,
            Body=payload,
            IfNoneMatch="*",
            ContentType="application/json",
            ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            Metadata={"sha256": digest},
        )
        if s3_readback_sha256(client, parsed.netloc, out_key) != digest:
            raise ValueError("S3 readback checksum mismatch for benchmark report")

        wandb.log({
            "pretrain_m16_proj_hours": m16["projected_100k_hours"],
            "pretrain_m32_proj_hours": m32["projected_100k_hours"],
            "grpo_steady_step_s": mean_total,
            "grpo_proj_1000_hours": report["grpo_v2"]["projected_1000_steps_hours"],
        })
        print("BENCHMARK REPORT VERIFIED:", json.dumps(report, indent=2), flush=True)
        failed = False
    finally:
        ray.shutdown()
        wandb.finish(exit_code=int(failed))


if __name__ == "__main__":
    main()
