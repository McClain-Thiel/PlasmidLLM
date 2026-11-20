"""CPU reward diagnostics on archived generations or real positive controls.

This checks implementation and reward sparsity; it is not a trained-v2 result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import wandb

from plasmid_llm.provenance import file_sha256
from plasmid_llm.utils import load_env_file
from post_training.scorers.plannotate_v2 import PlannotateV2Scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", required=True, help="LABEL=parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--registry", type=Path, default=Path("data/motif_registry_combined.parquet"))
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--require-token-prefix", default="<TAG_")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("sample-size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    panels = {}
    for item in args.panel:
        name, separator, path = item.partition("=")
        if not separator or not name or name in panels:
            raise ValueError("Panels must have unique labels and LABEL=path syntax")
        panels[name] = Path(path)
    root = Path(__file__).resolve().parents[2]
    load_env_file(root)
    scorer = PlannotateV2Scorer(motif_registry_path=str(args.registry))
    config = dict(sample_size=args.sample_size, selection="first matching rows",
                  require_token_prefix=args.require_token_prefix,
                  registry_sha256=file_sha256(args.registry),
                  scorer_sha256=file_sha256(root / "post_training/scorers/plannotate_v2.py"),
                  input_sha256={name: file_sha256(path) for name, path in panels.items()})
    with wandb.init(project="PlasmidLLM", name=args.run_name,
                    tags=["plasmidrl", "eval", "preflight"], config=config) as run:
        result = dict(config=config, wandb_url=run.url, panels={})
        for name, path in panels.items():
            panel = pd.read_parquet(path, columns=["prompt", "sequence"])
            mask = panel.prompt.str.contains(args.require_token_prefix, regex=False)
            selected = panel.loc[mask].head(args.sample_size)
            if selected.empty:
                raise ValueError(f"No matching prompts in panel {name}")
            start = time.monotonic()
            details = scorer.score_batch_detailed(selected.prompt.tolist(), selected.sequence.tolist())
            summary = {**scorer.last_diagnostics, "n": len(details),
                       "reward_mean": sum(r["reward"] for r in details) / len(details),
                       "elapsed_seconds": time.monotonic() - start}
            result["panels"][name] = dict(summary=summary, details=details,
                                         source_rows=selected.index.tolist())
            run.log({f"{name}/{key}": value for key, value in summary.items()})
            print(name, json.dumps(summary), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2)
        artifact = wandb.Artifact(f"{run.id}-reward-preflight", type="evaluation")
        artifact.add_file(str(args.output))
        artifact.add_file(__file__)
        run.log_artifact(artifact).wait()
        print(run.url)


if __name__ == "__main__":
    main()
