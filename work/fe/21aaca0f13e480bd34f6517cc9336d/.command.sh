#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AAEO01000042" \
    --bakta-gff "GenBank_AAEO01000042_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AAEO01000042_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AAEO01000042_copla_skip.tsv" \
    --output "GenBank_AAEO01000042_natural.json"
