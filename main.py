#!/usr/bin/env python3
"""Run the plasmid pretraining Metaflow pipeline."""

if __name__ == "__main__":
    from src.plasmid_pretraining.flow import PlasmidPretrainingFlow
    PlasmidPretrainingFlow()
