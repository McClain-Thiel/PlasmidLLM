#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AAEO01000040" \
    --bakta-gff "GenBank_AAEO01000040_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AAEO01000040_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AAEO01000040_copla_skip.tsv" \
    --output "GenBank_AAEO01000040_natural.json"
