from __future__ import annotations

from pyspark.sql import SparkSession

from .config import Settings


def get_spark(s: Settings | None = None, app: str = "lakehouse-agent") -> SparkSession:
    """Reuse the Databricks session if present; otherwise build a local one."""
    s = s or Settings()
    existing = SparkSession.getActiveSession()
    if existing is not None:
        return existing
    b = (
        SparkSession.builder.master("local[2]")
        .appName(app)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
    )
    if s.file_format == "delta":
        from delta import configure_spark_with_delta_pip  # needs Maven access for the jars

        b = b.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension").config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        return configure_spark_with_delta_pip(b).getOrCreate()
    return b.getOrCreate()
