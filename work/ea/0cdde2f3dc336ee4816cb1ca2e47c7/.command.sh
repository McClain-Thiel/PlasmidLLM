#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEK01000036.1" \
    --engineered "GenBank_AAEK01000036.1_engineered.json" \
    --metadata "GenBank_AAEK01000036.1_meta.json" \
    --natural "GenBank_AAEK01000036.1_natural.json" \
    --qc "GenBank_AAEK01000036.1_qc_skip.json" \
    --output "GenBank_AAEK01000036.1_classified.json"
