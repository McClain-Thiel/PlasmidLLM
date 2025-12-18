# SPACE Pipeline

**S**calable **P**lasmid **A**nnotation & **C**lassification **E**ngine

A high-throughput Nextflow pipeline that classifies plasmids as Natural or Engineered and extracts rich metadata into a unified "Golden Table" Parquet file.

## Data Source

Data is sourced from [PlasmidScope](https://plasmid.deepomics.org/) and [Addgene](https://www.addgene.org/), encompassing various plasmid types (mammalian, bacterial, etc.) and including metadata critical for conditioning generative models.

## Architecture

```
Raw Input (FASTA/GBK/JSON) → Normalization → [Parallel Tracks] → Classifier → Golden Table
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
    Track 1: Engineered  Track 2: Natural  Track 3: QC
    (PlasmidKit ori)     (Bakta/MOB/COPLA) (Complexity)
```

**Classification Logic**: If PlasmidKit detects a synthetic origin of replication → **Engineered**, else → **Natural**

## Supported Input Formats

| Format | Extensions | Metadata Richness | Notes |
|--------|------------|-------------------|-------|
| GenBank | `.gb`, `.gbk`, `.genbank` | Full | Best metadata extraction (topology, features, annotations) |
| Addgene JSON | `.json` | Full | Requires preprocessing with `split_addgene.py` |
| FASTA | `.fasta`, `.fa`, `.fna` | Minimal | Sequence only, limited metadata inference |

### Using Addgene Data

The pipeline supports Addgene's bulk plasmid JSON export. Due to its size (~167K plasmids), it must be split into individual files first:

```bash
# Split the large Addgene JSON into individual files
uv run src/nextflow/bin/split_addgene.py \
    --input ~/Downloads/addgene_plasmids.json \
    --output-dir data/addgene_split \
    --limit 1000  # Optional: limit for testing

# Run the pipeline on the split files
./nextflow run src/nextflow/main.nf -entry SIMPLE \
    --input_pattern "data/addgene_split/*.json" \
    --outdir results
```

**Addgene Field Mapping**:
| Addgene Field | Golden Table Field |
|---------------|-------------------|
| `bacterial_resistance` | `resistance_markers` |
| `plasmid_copy` | `copy_number` |
| `cloning.vector_types` | `plasmid_type` |
| `growth_strain` | `host` |
| `inserts[].tags` | `tags` |
| `inserts[].gene` | `reporter_genes` (if matched) |
| All other fields | `metadata.annotations` |

**Preserved Addgene Metadata** (in `metadata.annotations`):
- `addgene_id` - Original Addgene plasmid ID
- `url` - Link to Addgene page
- `sequence_source` - Where the sequence came from (addgene_full, user_full, etc.)
- `depositor_name`, `depositor_institution`, `pi_name` - Contributor info
- `article_references` - Associated publications
- `inserts` - Full insert details including species, Entrez gene IDs, cloning methods
- `cloning` - Backbone, vector types, primers
- `purpose` - Plasmid purpose/description

## Prerequisites

- Java 17+ (for Nextflow)
- Python 3.11+ with BioPython, pandas, pyarrow, plasmidkit
- `uv` for dependency management

## Installation

```bash
# Install Python dependencies
uv sync

# Install Nextflow (one-time)
curl -s https://get.nextflow.io | bash
```

## Usage

### Quick Start (SIMPLE mode - no external databases)

```bash
./nextflow run src/nextflow/main.nf -entry SIMPLE \
    --input_pattern "tests/data/*.gbk" \
    --outdir results
```

### Full Pipeline (with Bakta, MOB-suite, COPLA)

```bash
# Download databases first (one-time, ~30GB)
./nextflow run src/nextflow/main.nf -entry SETUP

# Run full pipeline
./nextflow run src/nextflow/main.nf \
    --input_pattern "data/*.gbk" \
    --outdir results
```

## Project Structure

```
src/nextflow/
├── main.nf                 # Main workflow entry point
├── nextflow.config         # Pipeline configuration
├── modules/
│   ├── normalize.nf        # Input normalization & metadata extraction
│   ├── scan_engineered.nf  # PlasmidKit synthetic origin detection
│   ├── scan_natural.nf     # Bakta + MOB-suite + COPLA annotation
│   ├── seq_qc.nf           # Sequence complexity metrics
│   ├── classify.nf         # Classification logic
│   ├── export.nf           # Parquet export
│   └── setup.nf            # Database download
├── bin/
│   ├── normalize.py        # GenBank/Addgene JSON parsing, metadata extraction
│   ├── split_addgene.py    # Preprocessing: split large Addgene JSON
│   ├── scan_engineered.py  # PlasmidKit wrapper
│   ├── scan_natural.py     # Natural annotation parser
│   ├── seq_qc.py           # QC metrics calculator
│   ├── classify.py         # Classification & merging
│   └── export_parquet.py   # Golden Table writer
└── containers/
    └── Dockerfile.space    # Custom container definition
```

## Output: Golden Table Schema

The final output is `{outdir}/final/plasmids.parquet` with the following schema:

### Primary Columns

| Column | Type | Description | Source Priority |
|--------|------|-------------|-----------------|
| `id` | UUID | Unique internal identifier | Generated |
| `seq_hash` | String | SHA256 hash of sequence (deduplication key) | Generated |
| `length` | Integer | Sequence length in base pairs | Calculated |
| `classification` | Enum | `Natural` or `Engineered` | PlasmidKit ori detection |
| `topology` | Enum | `circular` or `linear` | GenBank → Prediction → Default |
| `copy_number` | Enum | `high`, `medium`, `low`, `unknown` | GenBank → Origin mapping |
| `plasmid_type` | Enum | Expression system type | GenBank → Keyword matching |
| `host` | String | Expression host organism | GenBank → COPLA prediction |
| `origins` | List[String] | Detected synthetic origins | PlasmidKit |

### Nested JSON Columns

#### `features` - Track Results
```json
{
  "engineered": {
    "has_synthetic_ori": true,
    "origins": [{"name": "ColE1", "start": 100, "end": 500}],
    "markers": [{"name": "AmpR", "type": "resistance"}]
  },
  "natural": {
    "amr_genes": ["bla", "tetA"],
    "mobility": "conjugative",
    "replicon_type": "IncF",
    "predicted_host": "Escherichia",
    "ptu": "PTU-1"
  },
  "qc": {
    "gc_content": 0.52,
    "linguistic_complexity": 0.85,
    "synthesis_risk": 0.2,
    "max_homopolymer": 8
  }
}
```

#### `metadata` - GenBank Annotations
```json
{
  "organism": "Escherichia coli",
  "description": "Cloning vector pUC19",
  "original_id": "M77789.1",
  "resistance_markers": ["ampicillin"],
  "reporter_genes": ["lacZ"],
  "tags": ["his", "flag"],
  "cds_count": 3,
  "gene_count": 3,
  "genbank_features": [...],
  "annotations": {...}
}
```

## How Each Field is Derived

### Classification
- **Source**: PlasmidKit synthetic origin detector
- **Logic**: `has_synthetic_ori == true` → Engineered, else → Natural

### Topology
| Priority | Source | Method |
|----------|--------|--------|
| 1 | GenBank | `annotations.topology` field |
| 2 | Prediction | Synthetic ori detected → circular |
| 3 | Default | `circular` (most plasmids are circular) |

### Copy Number
| Priority | Source | Method |
|----------|--------|--------|
| 1 | GenBank | Keyword extraction from features/annotations |
| 2 | Origin mapping | ColE1/pUC/pBR322 → high, p15A → medium, pSC101 → low |
| 3 | Default | `unknown` |

**Origin → Copy Number Mapping**:
- **High copy** (~500-700): ColE1, pUC, pMB1, pBR322
- **Medium copy** (~15-20): p15A, pBBR1, pSA
- **Low copy** (~1-5): pSC101, F plasmid, RK2

### Plasmid Type
| Priority | Source | Method |
|----------|--------|--------|
| 1 | GenBank | Keyword matching in description/features |
| 2 | Prediction | Keyword matching on detected markers/promoters |
| 3 | Default | `unknown` |

**Type Keywords** (matched with word boundaries to avoid false positives):

| Type | Keywords |
|------|----------|
| `mammalian_expression` | mammalian, HEK, CHO, HeLa, COS, CMV, SV40, EF1a, CAG, PGK |
| `bacterial_expression` | bacterial, E. coli, T7, pLac, pTac, pAra, pTrc |
| `yeast_expression` | yeast, Saccharomyces, Pichia, GAL1, GAL10, ADH1 |
| `lentiviral` | lentivirus, lentiviral, LTR, psi packaging |
| `crispr` | CRISPR, Cas9, Cas12, gRNA, sgRNA |
| `cloning` | cloning, entry, Gateway, TOPO |

### Host
| Priority | Source | Method |
|----------|--------|--------|
| 1 | GenBank | `host` or `lab_host` qualifier in source feature |
| 2 | COPLA | Sequence-based host range prediction |
| 3 | Default | `unknown` |

### Resistance Markers, Reporter Genes, Tags
- **Source**: GenBank CDS/gene features
- **Method**: Keyword matching on `product`, `gene`, `note` qualifiers
- **Examples**:
  - Resistance: ampicillin, kanamycin, chloramphenicol, puromycin, blasticidin
  - Reporters: EGFP, mCherry, luciferase, lacZ
  - Tags: His, FLAG, Myc, HA, GST, MBP

## Pipeline Modules

| Module | Process | Description |
|--------|---------|-------------|
| `normalize.nf` | NORMALIZE | Extract metadata from GenBank, convert to FASTA |
| `scan_engineered.nf` | SCAN_ENGINEERED | PlasmidKit synthetic origin detection |
| `scan_natural.nf` | RUN_BAKTA, RUN_MOBSUITE, RUN_COPLA | Natural plasmid annotation |
| `seq_qc.nf` | SEQ_QC | Sequence complexity and synthesis risk |
| `classify.nf` | CLASSIFY | Merge tracks, apply classification logic |
| `export.nf` | EXPORT_PARQUET | Write Golden Table Parquet |

## Output Directory Structure

```
results/
├── normalized/          # Normalized FASTA files
├── metadata/            # Extracted GenBank metadata (JSON)
├── engineered_scan/     # PlasmidKit results
├── natural_scan/        # Bakta/MOB-suite/COPLA results
├── seq_qc/              # QC metrics
├── classified/          # Per-sample classified JSON
├── final/
│   ├── plasmids.parquet      # Golden Table
│   └── plasmids_summary.json # Summary statistics
└── pipeline_info/       # Nextflow reports
```
