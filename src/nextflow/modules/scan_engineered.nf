/*
 * SCAN_ENGINEERED: Track 1 - Engineered plasmid detection
 *
 * Uses PlasmidKit to detect synthetic origins of replication.
 * Simple classification: has_synthetic_ori -> Engineered
 */

process SCAN_ENGINEERED {
    tag "$sample_id"
    publishDir "${params.outdir}/engineered_scan", mode: 'copy'

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_engineered.json"), emit: json

    script:
    """
    scan_engineered.py \\
        --input "${fasta}" \\
        --output "${sample_id}_engineered.json" \\
        --sample-id "${sample_id}"
    """
}
