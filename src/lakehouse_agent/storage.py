"""One place that decides how a named table is written and read."""
from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession

from .config import Settings


def _path(s: Settings, layer: str, name: str) -> str:
    return os.path.join(s.root, layer, name)


def write_table(df: DataFrame, layer: str, name: str, s: Settings, mode: str = "overwrite") -> None:
    if s.storage_mode == "uc":
        df.write.format("delta").mode(mode).saveAsTable(f"{s.catalog}.{s.schema}.{layer}_{name}")
    else:
        df.write.format(s.file_format).mode(mode).save(_path(s, layer, name))


def read_table(spark: SparkSession, layer: str, name: str, s: Settings) -> DataFrame:
    if s.storage_mode == "uc":
        return spark.table(f"{s.catalog}.{s.schema}.{layer}_{name}")
    return spark.read.format(s.file_format).load(_path(s, layer, name))
