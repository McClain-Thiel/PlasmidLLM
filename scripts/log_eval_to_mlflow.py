"""Log plannotate eval results as notes/tags to MLflow runs.

Finds runs by checkpoint_dir tag, then logs eval summary metrics and notes.

Usage:
    python scripts/log_eval_to_mlflow.py
"""

import mlflow
import os

# ── Config ────────────────────────────────────────────────────────────────────

TRACKING_URI = "databricks"
EXPERIMENT_NAME = "/Users/mcclain.thiel@gmail.com/PlasmidLLM"

# Mapping from checkpoint dir basename to MLflow run name
CKPT_TO_RUN_NAME = {
    "transformer_15m_8k_eos": "transformer_15m_d384_seq8k_lr2e4_eos",
    "transformer_med_8k": "transformer_med_d512_seq8k",
    "transformer_conv_d384_seq8k_lr2e4": "transformer_conv_d384_seq8k_lr2e4",
    "transformer_conv_large_d512": "transformer_conv_large_d512_seq8k_lr2e4",
}

# Eval results: checkpoint_dir → summary stats
# From 20-sample plannotate eval (seed=42, temp=0.8, top_k=50, max_new_tokens=8000)
EVAL_RESULTS = {
    "checkpoints/transformer_15m_8k_eos": {
        "name": "transformer_15m_8k_eos",
        "n_samples": 20,
        "mean_overlap": 7.15,
        "mean_recall": 0.418,
        "total_overlap": 143,
        "total_true_features": 368,
        "macro_recall": "38.9%",
        "best_sample": "sample 40673: 18/21 features (86% recall) — mammalian expression vector",
        "worst_samples": "2 samples with 0 overlap (T5 promoter bacterial, spectinomycin+large)",
        "notes": (
            "Best overall model. Smallest (17.7M params) but highest feature recall (41.8%). "
            "Strong on common backbone features (lacI, ori, AmpR, T7). "
            "Consistent across vector types. All generations hit max 8000bp (no EOS termination observed in eval). "
            "Struggles with rare elements (PSC101, chloramphenicol, yeast-specific)."
        ),
    },
    "checkpoints/transformer_med_8k": {
        "name": "transformer_med_8k",
        "n_samples": 20,
        "mean_overlap": 5.65,
        "mean_recall": 0.316,
        "total_overlap": 113,
        "total_true_features": 368,
        "macro_recall": "30.7%",
        "best_sample": "sample 136180: 17/31 features (55% recall) — lentiviral vector",
        "worst_samples": "2 samples with 0 overlap (T5 bacterial, yeast vector)",
        "notes": (
            "Mid-size model (d384, 10L). Decent lentiviral recovery (sample 136180: 17/31). "
            "More variable than 15m_eos — some samples excellent, others collapse to 0. "
            "Good on Gateway cloning features (attL sites). "
            "Weaker than 15m_eos on mammalian expression vectors."
        ),
    },
    "checkpoints/transformer_conv_d384_seq8k_lr2e4": {
        "name": "transformer_conv_d384_seq8k_lr2e4",
        "n_samples": 20,
        "mean_overlap": 4.70,
        "mean_recall": 0.278,
        "total_overlap": 94,
        "total_true_features": 368,
        "macro_recall": "25.5%",
        "best_sample": "sample 140764: 14/19 features (74% recall) — bacterial T7 vector",
        "worst_samples": "4 samples with 0 overlap (large plasmid, yeast, mammalian, small bacterial)",
        "notes": (
            "Conv-augmented transformer (d384). Lowest feature recall overall. "
            "4 samples with zero overlap — highest failure rate. "
            "Strong on familiar bacterial backbone (sample 140764: 14/19), but poor generalization. "
            "Conv layers may need more training or different hyperparams."
        ),
    },
    "checkpoints/transformer_conv_large_d512": {
        "name": "transformer_conv_large_d512",
        "n_samples": 20,
        "mean_overlap": 6.15,
        "mean_recall": 0.329,
        "total_overlap": 123,
        "total_true_features": 368,
        "macro_recall": "33.4%",
        "best_sample": "sample 78089: 25/27 features (93% recall) — lentiviral CRISPR vector",
        "worst_samples": "3 samples with 0 overlap (CRISPR mammalian, yeast, synbio)",
        "notes": (
            "Largest conv model (d512). Highest single-sample score (25/27 = 93% on lentiviral CRISPR). "
            "Very inconsistent — 3 samples with 0 overlap. High variance suggests overfitting to common patterns. "
            "When it works, it works spectacularly (complex multi-element vectors). "
            "Needs more data or regularization to generalize."
        ),
    },
}


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"ERROR: Experiment '{EXPERIMENT_NAME}' not found")
        return

    experiment_id = experiment.experiment_id
    print(f"Experiment: {EXPERIMENT_NAME} (id={experiment_id})")

    # Search for all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="list",
    )

    print(f"Found {len(runs)} total runs\n")

    for ckpt_dir, eval_data in EVAL_RESULTS.items():
        basename = ckpt_dir.split("/")[-1]
        target_run_name = CKPT_TO_RUN_NAME.get(basename)

        # Find run by explicit run name mapping first
        matching = []
        if target_run_name:
            matching = [
                r for r in runs
                if (r.info.run_name or "") == target_run_name
            ]

        # Fallback: search by checkpoint_dir tag
        if not matching:
            matching = [
                r for r in runs
                if r.data.tags.get("checkpoint_dir", "").rstrip("/").endswith(basename)
            ]

        if not matching:
            print(f"WARNING: No run found for {ckpt_dir}")
            print(f"  Tried run_name='{target_run_name}' and checkpoint_dir containing '{basename}'")
            continue

        run = matching[0]
        run_id = run.info.run_id
        run_name = run.info.run_name
        print(f"Matched: {ckpt_dir} → run '{run_name}' (id={run_id})")

        with mlflow.start_run(run_id=run_id):
            # Log eval metrics
            mlflow.log_metrics({
                "eval_plannotate_mean_overlap": eval_data["mean_overlap"],
                "eval_plannotate_mean_recall": eval_data["mean_recall"],
                "eval_plannotate_total_overlap": eval_data["total_overlap"],
                "eval_plannotate_total_true_features": eval_data["total_true_features"],
                "eval_plannotate_n_samples": eval_data["n_samples"],
            })

            # Log eval summary as tags
            mlflow.set_tags({
                "eval_plannotate_notes": eval_data["notes"],
                "eval_plannotate_best_sample": eval_data["best_sample"],
                "eval_plannotate_worst_samples": eval_data["worst_samples"],
                "eval_plannotate_macro_recall": eval_data["macro_recall"],
            })

            # Set run description (note) with full summary
            description = (
                f"## Plannotate Eval (20 samples)\n\n"
                f"**Mean Feature Recall**: {eval_data['mean_recall']:.1%}\n"
                f"**Mean Overlap**: {eval_data['mean_overlap']:.1f} features\n"
                f"**Total**: {eval_data['total_overlap']}/{eval_data['total_true_features']} features recovered\n\n"
                f"**Best**: {eval_data['best_sample']}\n"
                f"**Worst**: {eval_data['worst_samples']}\n\n"
                f"**Notes**: {eval_data['notes']}"
            )
            mlflow.set_tag("mlflow.note.content", description)

        print(f"  Logged metrics + tags + description\n")

    print("Done!")


if __name__ == "__main__":
    main()
