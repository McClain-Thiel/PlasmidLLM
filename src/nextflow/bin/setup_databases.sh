#!/bin/bash
#
# setup_databases.sh - Download and setup databases for SPACE pipeline
#
# Usage:
#   ./setup_databases.sh [bakta|copla|all]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="${SCRIPT_DIR}/../databases"

mkdir -p "${DB_DIR}"

download_bakta() {
    echo "=== Downloading Bakta Database ==="
    echo "This may take a while (~30GB download)"

    BAKTA_DIR="${DB_DIR}/bakta"
    mkdir -p "${BAKTA_DIR}"

    # Check if Docker is available
    if command -v docker &> /dev/null; then
        echo "Using Docker to download Bakta DB..."
        docker run --rm -v "${BAKTA_DIR}:/db" oschwengers/bakta:latest \
            bakta_db download --output /db --type full
    else
        echo "Docker not available. Please install bakta and run:"
        echo "  bakta_db download --output ${BAKTA_DIR} --type full"
        exit 1
    fi

    echo "Bakta database downloaded to: ${BAKTA_DIR}"
}

download_copla() {
    echo "=== Setting up COPLA Database ==="

    COPLA_DIR="${DB_DIR}/copla"
    mkdir -p "${COPLA_DIR}"

    echo "COPLA database setup requires manual steps:"
    echo "1. Visit: https://github.com/santirdnd/COPLA"
    echo "2. Download the reference database"
    echo "3. Extract to: ${COPLA_DIR}"

    # Create placeholder
    touch "${COPLA_DIR}/.setup_required"

    echo "COPLA directory created at: ${COPLA_DIR}"
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  bakta   Download Bakta database (~30GB)"
    echo "  copla   Setup COPLA database (manual steps required)"
    echo "  all     Download all databases"
    echo "  help    Show this help message"
    echo ""
    echo "Database location: ${DB_DIR}"
}

case "${1:-help}" in
    bakta)
        download_bakta
        ;;
    copla)
        download_copla
        ;;
    all)
        download_bakta
        download_copla
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "=== Setup Complete ==="
echo "Database directory: ${DB_DIR}"
ls -la "${DB_DIR}"
