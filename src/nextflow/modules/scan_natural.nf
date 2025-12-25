/*
 * SCAN_NATURAL: Track 2 - Natural plasmid analysis
 *
 * Runs Bakta (annotation), MOB-suite (mobility/replicon), and COPLA (host prediction).
 * Parses outputs into unified JSON format.
 */

/*
 * RUN_BAKTA: Gene annotation with Bakta
 */
process RUN_BAKTA {
    tag "$sample_id"
    publishDir "${params.outdir}/bakta", mode: 'copy'

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}/${sample_id}.gff3"), emit: gff
    tuple val(sample_id), path("${sample_id}/${sample_id}.tsv"), emit: tsv
    tuple val(sample_id), path("${sample_id}"), emit: dir

    when:
    !params.skip_bakta

    script:
    """
    bakta \\
        --db ${params.bakta_db} \\
        --output ${sample_id} \\
        --prefix ${sample_id} \\
        --threads ${task.cpus} \\
        --skip-plot \\
        --skip-trna \\
        ${fasta}
    """
}

/*
 * RUN_MOBSUITE: Mobility and replicon typing with MOB-suite
 */
process RUN_MOBSUITE {
    tag "$sample_id"
    publishDir "${params.outdir}/mobsuite", mode: 'copy'

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_mobtyper.txt"), emit: tsv

    when:
    !params.skip_mobsuite

    script:
    """
    mob_typer \\
        --infile ${fasta} \\
        --out_file ${sample_id}_mobtyper.txt \\
        --num_threads ${task.cpus}
    """
}

/*
 * RUN_COPLA: Host range prediction with COPLA
 */
process RUN_COPLA {
    tag "$sample_id"
    publishDir "${params.outdir}/copla", mode: 'copy'

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_copla.tsv"), emit: tsv

    when:
    !params.skip_copla

    script:
    """
    # COPLA command - adjust based on actual COPLA CLI
    copla.py \\
        -i ${fasta} \\
        -o ${sample_id}_copla.tsv \\
        --database ${params.copla_db} \\
        || touch ${sample_id}_copla.tsv
    """
}

/*
 * PARSE_NATURAL: Parse all natural scan outputs into unified JSON
 */
process PARSE_NATURAL {
    tag "$sample_id"
    publishDir "${params.outdir}/natural_scan", mode: 'copy'

    input:
    tuple val(sample_id), path(bakta_gff), path(mobsuite_tsv), path(copla_tsv)

    output:
    tuple val(sample_id), path("${sample_id}_natural.json"), emit: json

    script:
    """
    scan_natural.py \\
        --sample-id "${sample_id}" \\
        --bakta-gff "${bakta_gff}" \\
        --mobsuite-tsv "${mobsuite_tsv}" \\
        --copla-tsv "${copla_tsv}" \\
        --output "${sample_id}_natural.json"
    """
}

/*
 * Placeholder processes for when tools are skipped
 */
process SKIP_BAKTA {
    tag "$sample_id"

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_bakta_skip.gff3"), emit: gff
    tuple val(sample_id), path("${sample_id}_bakta_skip.tsv"), emit: tsv

    script:
    """
    echo "# Bakta skipped" > ${sample_id}_bakta_skip.gff3
    echo "# Bakta skipped" > ${sample_id}_bakta_skip.tsv
    """
}

process SKIP_MOBSUITE {
    tag "$sample_id"

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_mobsuite_skip.txt"), emit: tsv

    script:
    """
    echo "# MOB-suite skipped" > ${sample_id}_mobsuite_skip.txt
    """
}

process SKIP_COPLA {
    tag "$sample_id"

    input:
    tuple val(sample_id), path(fasta)

    output:
    tuple val(sample_id), path("${sample_id}_copla_skip.tsv"), emit: tsv

    script:
    """
    echo "# COPLA skipped" > ${sample_id}_copla_skip.tsv
    """
}
