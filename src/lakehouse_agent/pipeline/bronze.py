"""Bronze: land raw source data as-is (all strings) plus lineage columns. Nothing is dropped."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


def _all_string(cols: list[str]) -> StructType:
    return StructType([StructField(c, StringType(), True) for c in cols])


def _with_lineage(df: DataFrame, batch_id: str) -> DataFrame:
    return (
        df.withColumn("_source_file", F.input_file_name())
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_batch_id", F.lit(batch_id))
    )


def ingest_customers(spark: SparkSession, path: str, batch_id: str) -> DataFrame:
    schema = _all_string(["customer_id", "full_name", "email", "region", "segment", "updated_at"])
    df = spark.read.option("header", True).schema(schema).csv(path)
    return _with_lineage(df, batch_id)


def ingest_invoices(spark: SparkSession, path: str, batch_id: str) -> DataFrame:
    schema = _all_string(["invoice_id", "customer_ref", "amount", "currency", "status", "issued_at"])
    return _with_lineage(spark.read.schema(schema).json(path), batch_id)


def ingest_tickets(spark: SparkSession, path: str, batch_id: str) -> DataFrame:
    schema = _all_string(["ticket_id", "customer_id", "subject", "priority", "status", "opened_at"])
    return _with_lineage(spark.read.schema(schema).json(path), batch_id)
