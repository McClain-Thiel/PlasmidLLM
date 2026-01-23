plasmid-pipeline

Databricks-ready pipeline scaffold for ingesting PLSDB FASTA files, curating silver tables, extracting features, and exporting ML training datasets. This replaces the previous repo layout with a notebook-first structure plus a small reusable library in `src/`.

Directory layout

- `config/`: Environment values and token thresholds
- `sql/`: Schema creation and optional optimization statements
- `notebooks/`: Orchestrated steps (bronze → silver → gold → platinum)
- `src/`: Reusable helpers (FASTA parsing, vocabulary, tagging, splits, Spark utils)
- `workflows/`: Optional job definition and task order

Getting started

- Local: review `config/env.yaml` and `config/token_config.yaml`, then open notebooks under `notebooks/` in your IDE or Databricks Connect environment
- Databricks: import the repo, update `config/env.yaml` for catalog/schemas and data paths, run notebooks in order (see `workflows/task_order.md`)

Key tables

- Bronze: raw sequences and auxiliary reference tables
- Silver: curated plasmid tables
- Gold: vocabulary and feature tables
- Platinum: tag assignments, augmented pairs, and exported training datasets

Notes

- This is a scaffold with minimal implementations. You may adapt functions in `src/` to your environment and data shapes.
