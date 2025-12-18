#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEN01000030.1" \
    --engineered "GenBank_AAEN01000030.1_engineered.json" \
    --metadata "GenBank_AAEN01000030.1_meta.json" \
    --natural "GenBank_AAEN01000030.1_natural.json" \
    --qc "GenBank_AAEN01000030.1_qc_skip.json" \
    --output "GenBank_AAEN01000030.1_classified.json"
