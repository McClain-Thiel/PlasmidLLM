#!/bin/bash
# Batch pLannotate evaluation across checkpoints and temperatures
# Run on g6-big

set -e

REPO=~/PlasmidLLM
CKPT_BASE=/opt/dlami/nvme/eval_checkpoints
OUTPUT_BASE=/opt/dlami/nvme/eval_output
PYTHON=/opt/dlami/nvme/plasmid_venv/bin/python
PLANNOTATE=/opt/dlami/nvme/miniconda3/envs/plannotate/bin/plannotate

N=50
BEST_OF=3
TEMPS="0.5 0.7 1.0"

CHECKPOINTS=(
    "grpo_plannotate/step_100"
    "grpo_plannotate/step_400"
    "grpo_plannotate/step_800"
    "grpo_dense_motif_v2/step_100"
    "grpo_dense_motif_v2/step_500"
    "grpo_dense_motif_v2/step_1000"
    "grpo_dense_motif_v2/final"
)

cd "$REPO"

for ckpt in "${CHECKPOINTS[@]}"; do
    for temp in $TEMPS; do
        ckpt_path="$CKPT_BASE/$ckpt"
        # Sanitize name for output dir
        name=$(echo "$ckpt" | tr '/' '_')
        out_dir="$OUTPUT_BASE/${name}_t${temp}"

        if [ -f "$out_dir/eval_report.md" ]; then
            echo "SKIP: $name temp=$temp (already done)"
            continue
        fi

        echo "=========================================="
        echo "Running: $name temp=$temp"
        echo "=========================================="

        $PYTHON -m evaluation.eval_plannotate \
            --models "$ckpt_path" \
            --n $N \
            --best-of $BEST_OF \
            --temperature $temp \
            --output-dir "$out_dir" \
            --plannotate-bin "$PLANNOTATE" \
            --motif-registry data/motif_registry.parquet \
            2>&1 | tee "$out_dir.log"

        echo ""
    done
done

echo "All evaluations complete!"
echo "Results in $OUTPUT_BASE/"
