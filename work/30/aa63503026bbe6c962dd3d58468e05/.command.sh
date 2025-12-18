#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AAEK01000038.1" \
    --bakta-gff "GenBank_AAEK01000038.1_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AAEK01000038.1_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AAEK01000038.1_copla_skip.tsv" \
    --output "GenBank_AAEK01000038.1_natural.json"
