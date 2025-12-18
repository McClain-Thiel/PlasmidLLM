# Plasmid Pretraining Dataset Pipeline

This project builds a pretraining dataset for plasmid backbone engineering from PlasmidScope data. It refactors original sequential scripts into a robust [Metaflow](https://docs.metaflow.org/) pipeline, enabling parallel processing and efficient data storage using Parquet.

## Data Source
Data is sourced from [PlasmidScope](https://plasmid.deepomics.org/), encompassing various plasmid types (mammalian, bacterial, etc.) and including metadata critical for conditioning generative models.

## Pipeline Architecture

The pipeline (`src/plasmid_pretraining/flow.py`) follows a **fan-out** pattern:

1.  **Start**: Discovers data sources (e.g., GenBank, RefSeq) from the input directory.
2.  **Process Source (Parallel)**: For each source, runs a sequential chain:
    *   **Extract**: Unpacks `.gbk` files from archives.
    *   **Parse**: Extracts sequences and metadata (resistance, copy number, tags). Saved as Parquet.
    *   **Annotate**: Identifies inserts (e.g., GFP, therapeutic genes) vs backbone sequences.
    *   **Create Pairs**: Generates prompt-response pairs for LLM training.
3.  **Split (Join)**: Aggregates all processed pairs, performs stratified 80/10/10 split (Train/Val/Test), and saves final artifacts.

## Usage

### Prerequisites
- Python 3.9+
- `uv` for dependency management

### Installation
```bash
uv sync
```

### Running the Pipeline

**Test Run (using provided test data):**
```bash
uv run python src/plasmid_pretraining/cli.py run --sources Test --input_dir ./tests/data --data_dir ./plasmid_data_output
```

**Full Run (assuming data in `plasmid_data`):**
```bash
uv run python src/plasmid_pretraining/cli.py run --data_dir ./plasmid_data
```

## Project Structure

*   `src/plasmid_pretraining/`: Core package
    *   `flow.py`: Metaflow pipeline definition
    *   `parsers.py`: GenBank parsing logic (BioPython)
    *   `annotators.py`: Insert detection logic
    *   `training.py`: Prompt engineering and tokenization
    *   `splitters.py`: Stratified splitting and file saving
    *   `constants.py`: Vocabulary and configuration
*   `tests/data/`: Test dataset (sample .gbk files)
*   `plasmid_data/`: (Generated) Output directory structure:
    *   `0_raw/`, `1_parsed/`, `2_annotated/`, `3_processed/`, `4_final/`

## Outputs

Final datasets are saved in `4_final/` in multiple formats:
*   `train.jsonl`, `val.jsonl`, `test.jsonl`: Simple prompt-response JSONL.
*   `train.parquet`, ...: Columnar format with full metadata.
*   `dataset_statistics.json`: Comprehensive report.
