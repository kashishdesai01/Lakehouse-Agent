"""Gold: business-facing tables the agent is allowed to query.

Definitions the agent's tool description repeats:
  revenue        = sum of amount_usd over invoices with status 'paid'
  total_billed   = sum of amount_usd over invoices that are not 'void'
  open ticket    = ticket with status 'open'
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def customer_360(customers: DataFrame, invoices: DataFrame, tickets: DataFrame) -> DataFrame:
    inv = invoices.groupBy("customer_id").agg(
        F.round(F.sum(F.when(F.col("status") != "void", F.col("amount_usd")).otherwise(0.0)), 2).alias("total_billed_usd"),
        F.round(F.sum(F.when(F.col("status") == "paid", F.col("amount_usd")).otherwise(0.0)), 2).alias("total_paid_usd"),
        F.count("*").alias("invoice_count"),
        F.max("issued_at").alias("last_invoice_at"),
    )
    tix = tickets.groupBy("customer_id").agg(
        F.sum(F.when(F.col("status") == "open", 1).otherwise(0)).alias("open_ticket_count"),
        F.sum(F.when((F.col("status") == "open") & (F.col("priority") == "P1"), 1).otherwise(0)).alias("open_p1_count"),
    )
    return (
        customers.select("customer_id", "full_name", "email", "region", "segment")
        .join(inv, "customer_id", "left")
        .join(tix, "customer_id", "left")
        .fillna({"total_billed_usd": 0.0, "total_paid_usd": 0.0, "invoice_count": 0,
                 "open_ticket_count": 0, "open_p1_count": 0})
    )


def monthly_revenue(customers: DataFrame, invoices: DataFrame) -> DataFrame:
    return (
        invoices.where("status = 'paid'")
        .join(customers.select("customer_id", "region", "segment"), "customer_id")
        .withColumn("month", F.date_format("issued_at", "yyyy-MM"))
        .groupBy("month", "region")
        .agg(F.round(F.sum("amount_usd"), 2).alias("revenue_usd"), F.count("*").alias("paid_invoice_count"))
    )


def ticket_summary(customers: DataFrame, tickets: DataFrame) -> DataFrame:
    return (
        tickets.join(customers.select("customer_id", "segment"), "customer_id")
        .groupBy("priority", "status", "segment")
        .agg(F.count("*").alias("ticket_count"))
    )
