#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEO01000042" \
    --engineered "GenBank_AAEO01000042_engineered.json" \
    --metadata "GenBank_AAEO01000042_meta.json" \
    --natural "GenBank_AAEO01000042_natural.json" \
    --qc "GenBank_AAEO01000042_qc_skip.json" \
    --output "GenBank_AAEO01000042_classified.json"
