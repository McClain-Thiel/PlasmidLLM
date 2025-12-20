/*
 * NORMALIZE: Input normalization and metadata extraction
 *
 * Takes FASTA/GenBank files and outputs:
 * - Normalized FASTA for downstream tools
 * - Rich metadata JSON with all GenBank annotations
 */

/*
 * SPLIT_JSON_FILE: Split large JSON files (like Addgene) into individual files
 */
process SPLIT_JSON_FILE {
    tag "$json_file.baseName"
    publishDir "${params.outdir}/split_json", mode: 'copy'

    input:
    path json_file

    output:
    path "split_output/*.json", emit: split_files

    script:
    """
    mkdir -p split_output
    split_addgene.py \\
        --input '${json_file}' \\
        --output-dir split_output
    """
}

process NORMALIZE {
    tag "$input_file.baseName"
    publishDir "${params.outdir}/normalized", mode: 'copy', pattern: '*.fasta'
    publishDir "${params.outdir}/metadata", mode: 'copy', pattern: '*_meta.json'
    errorStrategy 'ignore'  // Skip malformed files and continue

    input:
    path input_file

    output:
    tuple val(sample_id), path("${sample_id}.fasta"), emit: fasta
    tuple val(sample_id), path("${sample_id}_meta.json"), emit: meta

    script:
    sample_id = input_file.baseName.replaceAll(/\.(gb|gbk|fasta|fa)$/, '').replaceAll(/[^a-zA-Z0-9_.-]/, '_')
    """
    normalize.py \\
        --input '${input_file}' \\
        --output-fasta '${sample_id}.fasta' \\
        --output-meta '${sample_id}_meta.json'
    """
}
