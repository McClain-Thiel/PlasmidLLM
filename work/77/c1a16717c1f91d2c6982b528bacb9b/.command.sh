#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AAEO01000041" \
    --bakta-gff "GenBank_AAEO01000041_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AAEO01000041_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AAEO01000041_copla_skip.tsv" \
    --output "GenBank_AAEO01000041_natural.json"
