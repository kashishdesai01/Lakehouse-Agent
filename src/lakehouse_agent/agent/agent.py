"""Bounded tool-calling loop with a per-step trace. Tool failures are fed back, never raised to the user."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

from .llm import LLMClient
from .tools import TOOL_SPECS, ToolBox

SYSTEM_PROMPT = (
    "You answer questions about customers, billing and support using the tools provided. "
    "Use run_sql for numbers and search_documents for policy questions. Only state numbers that appear in tool "
    "results, and cite policy answers with the document id in [source: doc_id]. If the tools do not contain "
    "the answer, say so."
)


@dataclass
class Step:
    tool: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    hit_step_limit: bool = False


@contextmanager
def _span(name: str, inputs: dict):
    """MLflow span when tracing is available; a no-op otherwise. Errors from the body propagate unchanged."""
    cm = sp = None
    try:
        import mlflow

        cm = mlflow.start_span(name=name, span_type="TOOL")
        sp = cm.__enter__()
        sp.set_inputs(inputs)
    except Exception:  # tracing must never break the agent
        cm = sp = None
    try:
        yield sp
    except BaseException:
        if cm is not None:
            try:
                cm.__exit__(*sys.exc_info())
            except Exception:
                pass
        raise
    else:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass


class Agent:
    def __init__(self, llm: LLMClient, tools: ToolBox, max_steps: int = 6):
        self.llm, self.tools, self.max_steps = llm, tools, max_steps

    def ask(self, question: str) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        out = AgentResult(question=question, answer="")
        for _ in range(self.max_steps):
            turn = self.llm.next(messages, TOOL_SPECS)
            if "tool_calls" not in turn:
                out.answer = turn["content"]
                return out
            for call in turn["tool_calls"]:
                try:
                    with _span(call["name"], call["arguments"]) as sp:
                        result = self.tools.call(call["name"], call["arguments"])
                        if sp is not None:
                            sp.set_outputs(result)
                except Exception as e:  # SqlGuardError, Spark errors, bad tool names
                    result = {"error": f"{type(e).__name__}: {e}"}
                out.steps.append(Step(call["name"], call["arguments"], result))
                messages.append({"role": "tool", "name": call["name"], "tool_call_id": call.get("id", ""),
                                 "content": json.dumps(result, default=str)})
        out.hit_step_limit = True
        out.answer = "I could not finish within the step limit."
        return out
