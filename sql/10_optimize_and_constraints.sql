-- Run after tables exist; edit ZORDER cols to match your access patterns.

-- Example: sequences table
OPTIMIZE plasmids.plsdb_bronze.sequences ZORDER BY (nuccore_acc);

-- Gold features are often joined by accession
OPTIMIZE plasmids.plsdb_gold.plasmid_features_filtered ZORDER BY (accession);

-- Platinum outputs
OPTIMIZE plasmids.plsdb_platinum.tag_assignments ZORDER BY (accession);
OPTIMIZE plasmids.plsdb_platinum.sequences_joined ZORDER BY (accession);

-- Optional: add NOT NULL expectations via constraints if desired (Unity Catalog support varies by runtime)
-- ALTER TABLE ... ADD CONSTRAINT ...

