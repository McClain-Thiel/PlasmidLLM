#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AAEP01000051.1" \
    --bakta-gff "GenBank_AAEP01000051.1_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AAEP01000051.1_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AAEP01000051.1_copla_skip.tsv" \
    --output "GenBank_AAEP01000051.1_natural.json"
