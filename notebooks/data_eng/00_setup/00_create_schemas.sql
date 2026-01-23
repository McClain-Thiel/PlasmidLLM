CREATE CATALOG IF NOT EXISTS plasmids;
USE CATALOG plasmids;

CREATE SCHEMA IF NOT EXISTS plsdb_bronze COMMENT 'Raw PLSDB tables';
CREATE SCHEMA IF NOT EXISTS plsdb_silver COMMENT 'Cleaned/curated';
CREATE SCHEMA IF NOT EXISTS plsdb_gold COMMENT 'Feature engineering';
CREATE SCHEMA IF NOT EXISTS plsdb_platinum COMMENT 'ML-ready outputs';

