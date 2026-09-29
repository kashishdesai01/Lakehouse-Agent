"""Tools the agent can call. The SQL tool is read-only by construction, not by prompt wording."""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from ..config import GOLD_TABLES, Settings
from ..storage import read_table
from .retrieval import Retriever

GOLD_SCHEMA_DOC = """Tables (all amounts in USD):
customer_360(customer_id, full_name, email, region, segment, total_billed_usd, total_paid_usd,
  invoice_count, last_invoice_at, open_ticket_count, open_p1_count)
monthly_revenue(month 'YYYY-MM', region, revenue_usd, paid_invoice_count)
ticket_summary(priority 'P1'|'P2'|'P3', status 'open'|'closed'|'pending', segment, ticket_count)
Definitions: revenue = paid invoices only; total_billed excludes void invoices."""


class SqlGuardError(ValueError):
    pass


def validate_sql(query: str, allowed: tuple[str, ...] = GOLD_TABLES) -> str:
    """Accept exactly one SELECT over allow-listed tables; reject everything else."""
    try:
        parsed = sqlglot.parse(query, read="spark")
    except sqlglot.errors.ParseError as e:
        raise SqlGuardError(f"unparseable SQL: {e}") from e
    stmts = [p for p in parsed if p is not None]
    if len(stmts) != 1:
        raise SqlGuardError("exactly one statement is allowed")
    tree = stmts[0]
    if not isinstance(tree, (exp.Select, exp.With, exp.Union)) and not tree.find(exp.Select):
        raise SqlGuardError("only SELECT queries are allowed")
    if isinstance(tree, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Merge, exp.Alter,
                         exp.Command)):
        raise SqlGuardError("write or DDL statements are not allowed")
    if any(tree.find_all(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Merge, exp.Alter)):
        raise SqlGuardError("write or DDL statements are not allowed")
    cte_names = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if t.name in cte_names:
            continue
        if t.db or t.catalog:
            raise SqlGuardError("qualified table names are not allowed; use the bare gold table name")
        if t.name not in allowed:
            raise SqlGuardError(f"table '{t.name}' is not allowed; allowed: {', '.join(allowed)}")
    return tree.sql(dialect="spark")


class ToolBox:
    def __init__(self, spark, settings: Settings, retriever: Retriever):
        self.spark, self.s, self.retriever = spark, settings, retriever
        self._views_ready = False

    def _ensure_views(self) -> None:
        if not self._views_ready:
            for t in GOLD_TABLES:
                read_table(self.spark, "gold", t, self.s).createOrReplaceTempView(t)
            self._views_ready = True

    def run_sql(self, query: str) -> dict:
        safe = validate_sql(query)
        self._ensure_views()
        df = self.spark.sql(safe).limit(self.s.max_result_rows)
        rows = [r.asDict(recursive=True) for r in df.collect()]
        return {"columns": df.columns, "rows": rows, "row_count": len(rows), "sql": safe}

    def search_documents(self, query: str, k: int = 3) -> dict:
        hits = self.retriever.search(query, k=k)
        return {"results": hits}

    def call(self, name: str, args: dict) -> dict:
        if name == "run_sql":
            return self.run_sql(**args)
        if name == "search_documents":
            return self.search_documents(**args)
        raise ValueError(f"unknown tool: {name}")


TOOL_SPECS = [
    {"name": "run_sql", "description": "Run one read-only Spark SQL SELECT against the gold tables.\n" + GOLD_SCHEMA_DOC,
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "search_documents",
     "description": "Search policy and contract documents (refunds, SLA, retention, escalation, billing, security). "
                    "Returns chunks with doc_id to cite.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"]}},
]
