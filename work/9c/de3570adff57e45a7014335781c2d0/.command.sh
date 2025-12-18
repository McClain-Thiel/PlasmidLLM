#!/bin/bash -ue
classify.py \
    --sample-id "GenBank_AAEO01000040" \
    --engineered "GenBank_AAEO01000040_engineered.json" \
    --metadata "GenBank_AAEO01000040_meta.json" \
    --natural "GenBank_AAEO01000040_natural.json" \
    --qc "GenBank_AAEO01000040_qc_skip.json" \
    --output "GenBank_AAEO01000040_classified.json"
