# Anyscale post-training

`run_anyscale.py` is the entrypoint for GRPO post-training on managed Ray
compute. It loads one config from `post_training/configs`, downloads only the
data files referenced by that config, logs to W&B, and syncs checkpoints to
object storage.

## Required configuration

Create a gitignored `.env` in the repository root:

```bash
ANYSCALE_CLOUD=<cloud-name>
ANYSCALE_COMPUTE_CONFIG=<compute-config-name>
S3_DATA_PREFIX=s3://<bucket>/<prefix>
WANDB_API_KEY=<wandb-key>
HF_TOKEN=<hugging-face-token>
```

Optional variables:

```bash
CONFIG=grpo_plannotate_anyscale
ANYSCALE_JOB_NAME=grpo-plannotate
S3_CHECKPOINT_PREFIX=s3://<bucket>/<checkpoint-prefix>
RESUME_FROM_S3=s3://<bucket>/<checkpoint>
```

The data prefix must use this layout:

```text
<S3_DATA_PREFIX>/
├── data/
│   ├── training_pairs_v4.parquet
│   ├── motif_registry_combined.parquet
│   ├── plannotate_db.parquet
│   └── holdout_full_prompts.parquet  # required by v3 only
└── checkpoints/
```

Do not put credentials or account-specific paths in tracked YAML files.

## Submit

```bash
bash infra/anyscale/submit.sh
```

The script validates required variables before submission. The default config
is `post_training/configs/grpo_plannotate_anyscale.py`; select another with
`CONFIG`.

For a YAML-based submission, render the `${...}` placeholders in
`job_config.yaml` from the environment before passing it to the Anyscale CLI.
`compute_config.yaml` is a small single-GPU template; production compute
configs may instead be managed by name in Anyscale.

## Data and checkpoints

The entrypoint discovers repository-local data paths from the selected
`PostTrainingConfig` and downloads the corresponding filenames from
`<S3_DATA_PREFIX>/data/`. Existing local files are left untouched.

Checkpoints are written to the config's local checkpoint directory and synced
to:

```text
<S3_CHECKPOINT_PREFIX>/<config-name-without-_anyscale>/
```

Set `RESUME_FROM_S3` to replace the config's starting model with an existing
checkpoint.

## Monitoring

```bash
anyscale job list
anyscale job status --id <job-id>
anyscale job logs --id <job-id>
```

Training metrics, hyperparameters, and system metrics are logged to the W&B
project defined by the selected post-training config.
