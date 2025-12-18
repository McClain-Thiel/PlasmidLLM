"""CLI entry point for running the plasmid pretraining flow."""

import sys
from plasmid_pretraining.data_cleaning_flow import DataCleaningFlow


def main():
    """Run the Metaflow pipeline."""
    DataCleaningFlow()


if __name__ == "__main__":
    main()
