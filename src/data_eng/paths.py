from __future__ import annotations

from dataclasses import dataclass

from .settings import EnvConfig


@dataclass(frozen=True)
class TableNames:
    env: EnvConfig

    def fq(self, layer: str, table_key: str) -> str:
        """
        layer: one of bronze/silver/gold/platinum
        table_key: key in env.tables, e.g. 'bronze_sequences'
        """
        schema = self.env.schemas[layer]
        table = self.env.tables[table_key]
        return f"{self.env.catalog}.{schema}.{table}"

    def fq_bronze(self, table_key: str) -> str:
        return self.fq("bronze", table_key)

    def fq_silver(self, table_key: str) -> str:
        return self.fq("silver", table_key)

    def fq_gold(self, table_key: str) -> str:
        return self.fq("gold", table_key)

    def fq_platinum(self, table_key: str) -> str:
        return self.fq("platinum", table_key)

