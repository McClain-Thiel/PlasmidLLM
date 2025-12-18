/*
 * SETUP: Database download and setup processes
 *
 * One-time setup for external tool databases.
 */

/*
 * DOWNLOAD_BAKTA_DB: Download Bakta database (~30GB)
 */
process DOWNLOAD_BAKTA_DB {
    publishDir "${params.bakta_db}", mode: 'copy'

    output:
    path "db", emit: db_dir

    script:
    """
    # Download Bakta database
    bakta_db download --output db --type full

    echo "Bakta database downloaded successfully"
    """
}

/*
 * DOWNLOAD_COPLA_DB: Download COPLA database
 */
process DOWNLOAD_COPLA_DB {
    publishDir "${params.copla_db}", mode: 'copy'

    output:
    path "copla_db", emit: db_dir

    script:
    """
    # Download COPLA database (placeholder - adjust for actual COPLA)
    mkdir -p copla_db

    # COPLA typically needs:
    # - Reference plasmid sequences
    # - PTU (Plasmid Taxonomic Unit) database

    echo "COPLA database setup (manual download may be required)"
    echo "See: https://github.com/santirdnd/COPLA"

    touch copla_db/.placeholder
    """
}

/*
 * VERIFY_DATABASES: Check that required databases exist
 */
process VERIFY_DATABASES {
    input:
    val bakta_db_path
    val copla_db_path

    output:
    val true, emit: verified

    script:
    """
    #!/bin/bash
    set -e

    echo "Verifying databases..."

    # Check Bakta DB
    if [ -d "${bakta_db_path}" ]; then
        echo "Bakta DB found at: ${bakta_db_path}"
    else
        echo "WARNING: Bakta DB not found at: ${bakta_db_path}"
        echo "Run: nextflow run main.nf -entry SETUP_BAKTA"
    fi

    # Check COPLA DB
    if [ -d "${copla_db_path}" ]; then
        echo "COPLA DB found at: ${copla_db_path}"
    else
        echo "WARNING: COPLA DB not found at: ${copla_db_path}"
    fi

    echo "Database verification complete"
    """
}
