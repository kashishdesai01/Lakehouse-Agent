"""LLM interface + implementations.

`LLMClient.next(messages, tools)` returns either {"tool_calls": [{"name","arguments"}]} or
{"content": "final answer"}.

  * OpenAICompatibleLLM: talks to a Databricks Model Serving / Azure OpenAI style endpoint.
  * RuleBasedLLM: deterministic offline stand-in that routes by keyword. It exists to test orchestration,
    guardrails and the eval harness. Scores from it say nothing about real model quality.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol


class LLMClient(Protocol):
    def next(self, messages: list[dict], tools: list[dict]) -> dict: ...


class OpenAICompatibleLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or os.getenv("LLM_BASE_URL"),
                             api_key=api_key or os.getenv("LLM_API_KEY"))
        self.model = model

    def next(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, messages=messages,
            tools=[{"type": "function", "function": t} for t in tools])
        msg = resp.choices[0].message
        if msg.tool_calls:
            return {"tool_calls": [{"id": c.id, "name": c.function.name,
                                    "arguments": json.loads(c.function.arguments)} for c in msg.tool_calls]}
        return {"content": msg.content or ""}


_DOC_WORDS = re.compile(r"\b(policy|refund\w*|sla|response time|retain\w*|retention|escalat\w*|terms|encrypt\w*|security|due)\b", re.I)
_P = re.compile(r"\bP[123]\b")


class RuleBasedLLM:
    """Keyword router with fixed SQL templates. Deterministic; only for offline tests."""

    def next(self, messages: list[dict], tools: list[dict]) -> dict:
        q = next(m["content"] for m in messages if m["role"] == "user")
        results = [m for m in messages if m["role"] == "tool"]
        wants_doc, wants_sql = bool(_DOC_WORDS.search(q)), self._sql_for(q) is not None
        done_names = {m["name"] for m in results}
        calls = []
        if wants_sql and "run_sql" not in done_names:
            calls.append({"name": "run_sql", "arguments": {"query": self._sql_for(q)}})
        if wants_doc and "search_documents" not in done_names:
            calls.append({"name": "search_documents", "arguments": {"query": q}})
        if calls:
            return {"tool_calls": calls}
        return {"content": self._compose(results)}

    @staticmethod
    def _sql_for(q: str) -> str | None:
        ql = q.lower()
        p = _P.search(q)
        if "revenue" in ql and "region" in ql and ("highest" in ql or "top" in ql):
            return "SELECT region, ROUND(SUM(revenue_usd),2) AS revenue_usd FROM monthly_revenue GROUP BY region ORDER BY revenue_usd DESC LIMIT 1"
        if "total revenue" in ql:
            return "SELECT ROUND(SUM(revenue_usd),2) AS total_revenue_usd FROM monthly_revenue"
        if "open" in ql and "ticket" in ql and p:
            return f"SELECT SUM(ticket_count) AS open_tickets FROM ticket_summary WHERE status='open' AND priority='{p.group(0)}'"
        if "open" in ql and "ticket" in ql:
            return "SELECT SUM(ticket_count) AS open_tickets FROM ticket_summary WHERE status='open'"
        if "how many customers" in ql and "enterprise" in ql:
            return "SELECT COUNT(*) AS customers FROM customer_360 WHERE segment='enterprise'"
        if "how many customers" in ql:
            return "SELECT COUNT(*) AS customers FROM customer_360"
        if "most" in ql and "billed" in ql:
            return "SELECT customer_id, total_billed_usd FROM customer_360 ORDER BY total_billed_usd DESC LIMIT 1"
        return None

    @staticmethod
    def _compose(results: list[dict]) -> str:
        parts = []
        for m in results:
            data = json.loads(m["content"])
            if "error" in data:
                parts.append(f"I could not complete a step: {data['error']}")
            elif "rows" in data:
                for r in data["rows"]:
                    parts.append("; ".join(f"{k} = {v}" for k, v in r.items()))
            elif "results" in data and data["results"]:
                top = data["results"][0]
                parts.append(f"{top['text'].split(chr(10)*2, 1)[-1]} [source: {top['doc_id']}]")
            else:
                parts.append("No matching documents were found.")
        return " ".join(parts) or "I do not have enough information to answer."
