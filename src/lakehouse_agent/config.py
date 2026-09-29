"""Runtime settings. On Databricks, tables live in Unity Catalog; locally they are files."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Fixed FX table so revenue in gold is reproducible. A real deployment would join a rates table.
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}

GOLD_TABLES = ("customer_360", "monthly_revenue", "ticket_summary")


@dataclass
class Settings:
    # "uc" = Unity Catalog managed tables (Databricks). "path" = files under root (local/tests).
    storage_mode: str = field(default_factory=lambda: os.getenv("LH_STORAGE_MODE", "path"))
    # File format used in "path" mode. "delta" needs delta-spark jars; "parquet" runs anywhere.
    file_format: str = field(default_factory=lambda: os.getenv("LH_FILE_FORMAT", "parquet"))
    root: str = field(default_factory=lambda: os.getenv("LH_ROOT", ".lakehouse"))
    catalog: str = field(default_factory=lambda: os.getenv("LH_CATALOG", "main"))
    schema: str = field(default_factory=lambda: os.getenv("LH_SCHEMA", "lakehouse_agent"))
    max_result_rows: int = 200


def add_settings_args(parser) -> None:
    parser.add_argument("--storage-mode", choices=["path", "uc"], default=None)
    parser.add_argument("--file-format", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--schema", default=None)


def settings_from_args(args) -> "Settings":
    s = Settings()
    for k in ("storage_mode", "file_format", "root", "catalog", "schema"):
        v = getattr(args, k, None)
        if v:
            setattr(s, k, v)
    return s
