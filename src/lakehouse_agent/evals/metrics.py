"""Deterministic scoring. Hallucination here means: the answer states a number or cites a document
that none of the tool results support. It does not judge reasoning quality or tone."""
from __future__ import annotations

import json
import re

NUM = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
CITE = re.compile(r"\[source:\s*([\w\-]+)\]")
ABSTAIN_MARKERS = ("do not have", "don't have", "cannot", "can't", "could not", "not able", "no matching",
                   "not allowed", "no information", "unable")


def _floats(text: str) -> list[float]:
    return [float(t.replace(",", "")) for t in NUM.findall(text)]


def _walk(obj, out: list[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(_floats(obj))


def tool_evidence(steps) -> tuple[list[float], set[str]]:
    nums: list[float] = []
    docs: set[str] = set()
    for s in steps:
        if "error" in s.result:
            continue
        _walk(s.result, nums)
        for r in s.result.get("results", []):
            docs.add(r["doc_id"])
    return nums, docs


def ungrounded(answer: str, question: str, steps) -> dict:
    nums, docs = tool_evidence(steps)
    allowed = nums + _floats(question)
    bad_numbers = [n for n in _floats(CITE.sub("", answer))
                   if not any(abs(n - a) < 0.011 for a in allowed)]
    bad_cites = [c for c in CITE.findall(answer) if c not in docs]
    return {"numbers": bad_numbers, "citations": bad_cites}


def value_present(answer: str, expected: str) -> bool:
    exp_nums = _floats(expected)
    if exp_nums and expected.replace(",", "").replace(".", "").isdigit():
        return any(abs(a - exp_nums[0]) < 0.011 for a in _floats(answer))
    a = re.sub(r"\s+", " ", answer.lower())
    e = re.sub(r"\s+", " ", expected.lower())
    if re.fullmatch(r"\d+ \w+", e):  # "45 days": require number and unit to be adjacent
        return e in a
    return e in a


def is_abstention(answer: str) -> bool:
    return any(m in answer.lower() for m in ABSTAIN_MARKERS)


def retrieved_docs(steps) -> list[str]:
    out: list[str] = []
    for s in steps:
        if s.tool == "search_documents" and "results" in s.result:
            out.extend(r["doc_id"] for r in s.result["results"])
    return out


def score_item(item: dict, result) -> dict:
    ans = result.answer
    kind = item["type"]
    docs = retrieved_docs(result.steps)
    bad = ungrounded(ans, item["question"], result.steps)
    errors = sum(1 for s in result.steps if "error" in s.result)
    row = {"id": item["id"], "type": kind, "question": item["question"], "answer": ans,
           "steps": len(result.steps), "tool_errors": errors, "ungrounded": bad,
           "hallucinated": bool(bad["numbers"] or bad["citations"]), "hit_step_limit": result.hit_step_limit}
    if kind == "abstain":
        row["correct"] = is_abstention(ans) and not bad["numbers"]
    else:
        row["correct"] = all(value_present(ans, v) for v in item["expected_values"])
    if item.get("expected_doc"):
        row["retrieval_hit"] = item["expected_doc"] in docs
        row["retrieval_top1"] = bool(docs) and docs[0] == item["expected_doc"]
    return row


def summarize(rows: list[dict]) -> dict:
    def rate(pred, subset):
        s = [r for r in rows if subset(r)]
        return round(sum(1 for r in s if pred(r)) / len(s), 4) if s else None

    return {
        "n_questions": len(rows),
        "answer_accuracy": rate(lambda r: r["correct"], lambda r: True),
        "answer_accuracy_sql": rate(lambda r: r["correct"], lambda r: r["type"] == "sql"),
        "answer_accuracy_doc": rate(lambda r: r["correct"], lambda r: r["type"] == "doc"),
        "answer_accuracy_hybrid": rate(lambda r: r["correct"], lambda r: r["type"] == "hybrid"),
        "abstain_accuracy": rate(lambda r: r["correct"], lambda r: r["type"] == "abstain"),
        "retrieval_hit_rate": rate(lambda r: r["retrieval_hit"], lambda r: "retrieval_hit" in r),
        "retrieval_top1_rate": rate(lambda r: r["retrieval_top1"], lambda r: "retrieval_top1" in r),
        "hallucination_rate": rate(lambda r: r["hallucinated"], lambda r: True),
        "tool_error_rate": rate(lambda r: r["tool_errors"] > 0, lambda r: True),
        "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 3) if rows else None,
    }
