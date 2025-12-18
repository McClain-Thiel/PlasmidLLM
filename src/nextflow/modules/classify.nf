/*
 * CLASSIFY: Merge results and classify plasmids
 *
 * Combines outputs from all tracks and applies classification logic:
 * - Simple: has_synthetic_ori -> Engineered, else Natural
 * - Primary columns: GenBank > Prediction precedence
 */

process CLASSIFY {
    tag "$sample_id"
    publishDir "${params.outdir}/classified", mode: 'copy'

    input:
    tuple val(sample_id), path(engineered_json), path(meta_json), path(natural_json), path(qc_json)

    output:
    tuple val(sample_id), path("${sample_id}_classified.json"), emit: json

    script:
    """
    classify.py \\
        --sample-id "${sample_id}" \\
        --engineered "${engineered_json}" \\
        --metadata "${meta_json}" \\
        --natural "${natural_json}" \\
        --qc "${qc_json}" \\
        --output "${sample_id}_classified.json"
    """
}
