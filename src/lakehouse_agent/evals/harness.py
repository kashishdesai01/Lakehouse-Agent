"""Run the agent over the golden set, score it, and log the run to MLflow."""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from ..agent.agent import Agent
from .metrics import score_item, summarize


def load_golden(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(agent: Agent, golden: list[dict], experiment: str | None = None, run_name: str | None = None,
             tracking_uri: str | None = None, params: dict | None = None) -> dict:
    """Return {"summary": {...}, "rows": [...]}. Logs to MLflow when `experiment` is given."""
    def _run() -> dict:
        rows = []
        for item in golden:
            res = agent.ask(item["question"])
            rows.append(score_item(item, res))
        return {"summary": summarize(rows), "rows": rows}

    if not experiment:
        return _run()

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"llm": type(agent.llm).__name__, "max_steps": agent.max_steps,
                           "n_golden": len(golden), **(params or {})})
        out = _run()
        mlflow.log_metrics({k: v for k, v in out["summary"].items() if isinstance(v, (int, float))})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "eval_rows.json")
            with open(p, "w") as f:
                json.dump(out["rows"], f, indent=2, default=str)
            mlflow.log_artifact(p)
        return out


def main() -> None:
    from ..agent.llm import OpenAICompatibleLLM, RuleBasedLLM
    from ..agent.retrieval import TfidfRetriever, load_chunks
    from ..agent.tools import ToolBox
    from ..config import add_settings_args, settings_from_args
    from ..spark_session import get_spark

    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--llm", choices=["rule", "openai"], default="rule")
    p.add_argument("--model", default="")
    p.add_argument("--experiment", default="lakehouse-agent-eval")
    add_settings_args(p)
    a = p.parse_args()
    s = settings_from_args(a)
    spark = get_spark(s)
    tools = ToolBox(spark, s, TfidfRetriever(load_chunks(f"{a.raw_dir}/docs")))
    llm = RuleBasedLLM() if a.llm == "rule" else OpenAICompatibleLLM(model=a.model)
    out = evaluate(Agent(llm, tools), load_golden(a.golden), experiment=a.experiment)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
