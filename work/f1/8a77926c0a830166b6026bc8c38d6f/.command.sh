#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AAEN01000030.1" \
    --bakta-gff "GenBank_AAEN01000030.1_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AAEN01000030.1_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AAEN01000030.1_copla_skip.tsv" \
    --output "GenBank_AAEN01000030.1_natural.json"
