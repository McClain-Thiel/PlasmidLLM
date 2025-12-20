#!/usr/bin/env nextflow

/*
 * SPACE Pipeline: Scalable Plasmid Annotation & Classification Engine
 *
 * A high-throughput pipeline to ingest, classify, and annotate plasmid sequences.
 *
 * Architecture:
 *   Raw Input -> Normalization -> [Parallel Tracks] -> Classifier -> Golden Table
 *                                       |
 *               +---------------+-------+-------+
 *               |               |               |
 *        Track 1: Engineered  Track 2: Natural  Track 3: QC
 *        (PlasmidKit ori)    (Bakta/MOB/COPLA) (Complexity)
 *
 * Classification: has_synthetic_ori -> Engineered, else Natural
 */

nextflow.enable.dsl=2

// Include modules
include { NORMALIZE; SPLIT_JSON_FILE } from './modules/normalize'
include { SCAN_ENGINEERED } from './modules/scan_engineered'
include { RUN_BAKTA; RUN_MOBSUITE; RUN_COPLA; PARSE_NATURAL; SKIP_BAKTA; SKIP_MOBSUITE; SKIP_COPLA } from './modules/scan_natural'
include { SEQ_QC; SKIP_QC } from './modules/seq_qc'
include { CLASSIFY } from './modules/classify'
include { EXPORT_PARQUET } from './modules/export'
include { DOWNLOAD_BAKTA_DB; DOWNLOAD_COPLA_DB; VERIFY_DATABASES } from './modules/setup'

// Log pipeline info
log.info """
===========================================
 S P A C E   P I P E L I N E   v${workflow.manifest.version ?: '0.1.0'}
===========================================
Scalable Plasmid Annotation & Classification Engine

Input pattern : ${params.input_pattern}
Output dir    : ${params.outdir}
Profile       : ${workflow.profile}

Skip Bakta    : ${params.skip_bakta}
Skip MOB-suite: ${params.skip_mobsuite}
Skip COPLA    : ${params.skip_copla}
Skip QC       : ${params.skip_qc}
===========================================
"""

/*
 * Main workflow
 */
workflow {
    // Input channel from file pattern(s) - supports list or single pattern
    if (params.input_pattern instanceof List) {
        raw_input_ch = Channel.fromPath(params.input_pattern, checkIfExists: true)
    } else {
        raw_input_ch = Channel.fromPath(params.input_pattern, checkIfExists: true)
    }

    // Branch inputs: JSONs need splitting, others proceed directly
    input_branch = raw_input_ch.branch {
        json: it.name.endsWith('.json')
        other: true
    }

    // Split large JSONs
    SPLIT_JSON_FILE(input_branch.json)
    
    // Flatten the split output (list of files) into individual items
    split_json_ch = SPLIT_JSON_FILE.out.split_files.flatten()

    // Merge original non-JSON files with the newly split JSON files
    // And map to (id, file) tuple
    input_ch = input_branch.other.mix(split_json_ch)
        .map { file -> tuple(file.baseName.replaceAll(/\.(gb|gbk|fasta|fa|json)$/, '').replaceAll(/[^a-zA-Z0-9_.-]/, '_'), file) }

    // Log input files
    input_ch.count().view { n -> "Processing ${n} input files" }

    // Step 1: Normalization + Metadata extraction
    NORMALIZE(input_ch.map { id, file -> file })

    // Step 2: Parallel tracks

    // Track 1: Engineered scan (PlasmidKit ori detection)
    SCAN_ENGINEERED(NORMALIZE.out.fasta)

    // Track 2: Natural scan (Bakta + MOB-suite + COPLA)
    if (params.skip_bakta) {
        SKIP_BAKTA(NORMALIZE.out.fasta)
        bakta_gff = SKIP_BAKTA.out.gff
    } else {
        RUN_BAKTA(NORMALIZE.out.fasta)
        bakta_gff = RUN_BAKTA.out.gff
    }

    if (params.skip_mobsuite) {
        SKIP_MOBSUITE(NORMALIZE.out.fasta)
        mobsuite_tsv = SKIP_MOBSUITE.out.tsv
    } else {
        RUN_MOBSUITE(NORMALIZE.out.fasta)
        mobsuite_tsv = RUN_MOBSUITE.out.tsv
    }

    if (params.skip_copla) {
        SKIP_COPLA(NORMALIZE.out.fasta)
        copla_tsv = SKIP_COPLA.out.tsv
    } else {
        RUN_COPLA(NORMALIZE.out.fasta)
        copla_tsv = RUN_COPLA.out.tsv
    }

    // Join natural scan outputs by sample ID
    natural_inputs = bakta_gff
        .join(mobsuite_tsv)
        .join(copla_tsv)

    PARSE_NATURAL(natural_inputs)

    // Track 3: Sequence QC
    if (params.skip_qc) {
        SKIP_QC(NORMALIZE.out.fasta)
        qc_json = SKIP_QC.out.json
    } else {
        SEQ_QC(NORMALIZE.out.fasta)
        qc_json = SEQ_QC.out.json
    }

    // Step 3: Join all track results by sample ID
    // Format: (sample_id, engineered_json, meta_json, natural_json, qc_json)
    classify_inputs = SCAN_ENGINEERED.out.json
        .join(NORMALIZE.out.meta)
        .join(PARSE_NATURAL.out.json)
        .join(qc_json)

    // Step 4: Classify
    CLASSIFY(classify_inputs)

    // Step 5: Export to Parquet Golden Table
    classified_files = CLASSIFY.out.json.map { id, file -> file }.collect()
    EXPORT_PARQUET(classified_files)
}

/*
 * Setup workflow: Download databases
 */
workflow SETUP_BAKTA {
    DOWNLOAD_BAKTA_DB()
}

workflow SETUP_COPLA {
    DOWNLOAD_COPLA_DB()
}

workflow SETUP {
    DOWNLOAD_BAKTA_DB()
    DOWNLOAD_COPLA_DB()
}

/*
 * Simplified workflow: Skip external tools (Bakta, MOB-suite, COPLA)
 * Useful for quick testing or when databases aren't available
 */
workflow SIMPLE {
    // Input channel
    input_ch = Channel.fromPath(params.input_pattern, checkIfExists: true)
        .map { file -> tuple(file.baseName.replaceAll(/\.(gb|gbk|fasta|fa)$/, '').replaceAll(/[^a-zA-Z0-9_.-]/, '_'), file) }

    input_ch.count().view { n -> "Processing ${n} input files (SIMPLE mode)" }

    // Normalization
    NORMALIZE(input_ch.map { id, file -> file })

    // Track 1: Engineered scan only
    SCAN_ENGINEERED(NORMALIZE.out.fasta)

    // Skip Track 2 & 3 - use placeholder data
    SKIP_BAKTA(NORMALIZE.out.fasta)
    SKIP_MOBSUITE(NORMALIZE.out.fasta)
    SKIP_COPLA(NORMALIZE.out.fasta)
    SKIP_QC(NORMALIZE.out.fasta)

    natural_inputs = SKIP_BAKTA.out.gff
        .join(SKIP_MOBSUITE.out.tsv)
        .join(SKIP_COPLA.out.tsv)

    PARSE_NATURAL(natural_inputs)

    // Classify with available data
    classify_inputs = SCAN_ENGINEERED.out.json
        .join(NORMALIZE.out.meta)
        .join(PARSE_NATURAL.out.json)
        .join(SKIP_QC.out.json)

    CLASSIFY(classify_inputs)

    // Export
    classified_files = CLASSIFY.out.json.map { id, file -> file }.collect()
    EXPORT_PARQUET(classified_files)
}

/*
 * Completion handler
 */
workflow.onComplete {
    log.info """
===========================================
Pipeline completed!
===========================================
Status    : ${workflow.success ? 'SUCCESS' : 'FAILED'}
Duration  : ${workflow.duration}
Output    : ${params.outdir}

Results:
  - Normalized:  ${params.outdir}/normalized/
  - Engineered:  ${params.outdir}/engineered_scan/
  - Natural:     ${params.outdir}/natural_scan/
  - QC:          ${params.outdir}/seq_qc/
  - Classified:  ${params.outdir}/classified/
  - Final:       ${params.outdir}/final/plasmids.parquet
===========================================
"""
}

workflow.onError {
    log.error "Pipeline failed: ${workflow.errorMessage}"
}
