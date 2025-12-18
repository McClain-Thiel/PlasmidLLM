#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEK01000038.1" \
    --engineered "GenBank_AAEK01000038.1_engineered.json" \
    --metadata "GenBank_AAEK01000038.1_meta.json" \
    --natural "GenBank_AAEK01000038.1_natural.json" \
    --qc "GenBank_AAEK01000038.1_qc_skip.json" \
    --output "GenBank_AAEK01000038.1_classified.json"
