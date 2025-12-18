#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEP01000051.1" \
    --engineered "GenBank_AAEP01000051.1_engineered.json" \
    --metadata "GenBank_AAEP01000051.1_meta.json" \
    --natural "GenBank_AAEP01000051.1_natural.json" \
    --qc "GenBank_AAEP01000051.1_qc_skip.json" \
    --output "GenBank_AAEP01000051.1_classified.json"
