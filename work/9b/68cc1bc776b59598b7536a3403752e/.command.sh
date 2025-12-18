#!/bin/bash -ue
export_parquet.py \
    --input-dir . \
    --output plasmids.parquet \
    --summary plasmids_summary.json
