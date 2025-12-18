#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AADR01000058.1" \
    --engineered "GenBank_AADR01000058.1_engineered.json" \
    --metadata "GenBank_AADR01000058.1_meta.json" \
    --natural "GenBank_AADR01000058.1_natural.json" \
    --qc "GenBank_AADR01000058.1_qc_skip.json" \
    --output "GenBank_AADR01000058.1_classified.json"
