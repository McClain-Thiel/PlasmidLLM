#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEO01000041" \
    --engineered "GenBank_AAEO01000041_engineered.json" \
    --metadata "GenBank_AAEO01000041_meta.json" \
    --natural "GenBank_AAEO01000041_natural.json" \
    --qc "GenBank_AAEO01000041_qc_skip.json" \
    --output "GenBank_AAEO01000041_classified.json"
