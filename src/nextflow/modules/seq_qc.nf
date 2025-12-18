/*
 * SEQ_QC: Track 3 - Sequence quality and complexity metrics
 *
 * Calculates GC content, homopolymers, repeats, and synthesis risk score.
 */

process SEQ_QC {
    tag "$sample_id"
    publishDir "${params.outdir}/seq_qc", mode: 'copy'

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_qc.json"), emit: json

    when:
    !params.skip_qc

    script:
    """
    seq_qc.py \\
        --input "${fasta}" \\
        --output "${sample_id}_qc.json" \\
        --sample-id "${sample_id}"
    """
}

/*
 * Placeholder when QC is skipped
 */
process SKIP_QC {
    tag "$sample_id"

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_qc_skip.json"), emit: json

    script:
    """
    echo '{"sample_id": "${sample_id}", "skipped": true}' > ${sample_id}_qc_skip.json
    """
}
