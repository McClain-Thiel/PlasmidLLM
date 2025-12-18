#!/bin/bash -ue
scan_natural.py \
    --sample-id "GenBank_AADR01000058.1" \
    --bakta-gff "GenBank_AADR01000058.1_bakta_skip.gff3" \
    --mobsuite-tsv "GenBank_AADR01000058.1_mobsuite_skip.txt" \
    --copla-tsv "GenBank_AADR01000058.1_copla_skip.tsv" \
    --output "GenBank_AADR01000058.1_natural.json"
