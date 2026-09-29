"""Silver: typed, cleaned, deduplicated records. Rule failures go to quarantine, not the void."""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ..config import FX_TO_USD
from .expectations import CUSTOMER_RULES, INVOICE_RULES, TICKET_RULES, split_valid


def _latest(df: DataFrame, key: str, order_col: str) -> DataFrame:
    w = Window.partitionBy(key).orderBy(F.col(order_col).desc_nulls_last(), F.col("_ingested_at").desc())
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def type_customers(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        F.upper(F.trim("customer_id")).alias("customer_id"),
        F.initcap(F.trim("full_name")).alias("full_name"),
        F.lower(F.trim("email")).alias("email"),
        F.upper(F.trim("region")).alias("region"),
        F.lower(F.trim("segment")).alias("segment"),
        F.to_timestamp("updated_at").alias("updated_at"),
        "_source_file", "_ingested_at", "_batch_id",
    )


def clean_customers(bronze: DataFrame) -> tuple[DataFrame, DataFrame]:
    typed = type_customers(bronze)
    valid, quarantined = split_valid(typed, CUSTOMER_RULES)
    return _latest(valid, "customer_id", "updated_at"), quarantined


def type_invoices(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        F.trim("invoice_id").alias("invoice_id"),
        # billing refers to customers as "0001"; CRM and support use "C0001".
        F.concat(F.lit("C"), F.lpad(F.trim("customer_ref"), 4, "0")).alias("customer_id"),
        F.col("amount").cast("double").alias("amount"),
        F.upper(F.trim("currency")).alias("currency"),
        F.lower(F.trim("status")).alias("status"),
        F.to_timestamp("issued_at").alias("issued_at"),
        "_source_file", "_ingested_at", "_batch_id",
    )


def add_usd(df: DataFrame) -> DataFrame:
    fx = F.create_map(*[x for k, v in FX_TO_USD.items() for x in (F.lit(k), F.lit(v))])
    return df.withColumn("amount_usd", F.round(F.col("amount") * fx[F.col("currency")], 2))


def clean_invoices(bronze: DataFrame, customers: DataFrame) -> tuple[DataFrame, DataFrame]:
    typed = type_invoices(bronze)
    valid, quarantined = split_valid(typed, INVOICE_RULES)
    deduped = _latest(valid, "invoice_id", "issued_at")
    known = deduped.join(customers.select("customer_id"), "customer_id", "left_semi")
    orphans = (
        deduped.join(customers.select("customer_id"), "customer_id", "left_anti")
        .withColumn("_failed_rules", F.lit("customer_exists"))
    )
    return add_usd(known), quarantined.unionByName(orphans, allowMissingColumns=True)


def type_tickets(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        F.trim("ticket_id").alias("ticket_id"),
        F.upper(F.trim("customer_id")).alias("customer_id"),
        F.trim("subject").alias("subject"),
        F.upper(F.trim("priority")).alias("priority"),
        F.lower(F.trim("status")).alias("status"),
        F.to_timestamp("opened_at").alias("opened_at"),
        "_source_file", "_ingested_at", "_batch_id",
    )


def clean_tickets(bronze: DataFrame, customers: DataFrame) -> tuple[DataFrame, DataFrame]:
    typed = type_tickets(bronze)
    valid, quarantined = split_valid(typed, TICKET_RULES)
    deduped = _latest(valid, "ticket_id", "opened_at")
    known = deduped.join(customers.select("customer_id"), "customer_id", "left_semi")
    orphans = (
        deduped.join(customers.select("customer_id"), "customer_id", "left_anti")
        .withColumn("_failed_rules", F.lit("customer_exists"))
    )
    return known, quarantined.unionByName(orphans, allowMissingColumns=True)
