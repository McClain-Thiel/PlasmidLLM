/*
 * EXPORT_PARQUET: Export classified results to Parquet Golden Table
 *
 * Collects all classified JSON files and creates unified Parquet output.
 */

process EXPORT_PARQUET {
    publishDir "${params.outdir}/final", mode: 'copy'

    input:
    path classified_jsons

    output:
    path "plasmids.parquet", emit: parquet
    path "plasmids_summary.json", emit: summary

    script:
    """
    export_parquet.py \\
        --input-dir . \\
        --output plasmids.parquet \\
        --summary plasmids_summary.json
    """
}
