import json

from lakehouse_agent.agent.agent import Agent
from lakehouse_agent.agent.llm import RuleBasedLLM
from lakehouse_agent.agent.retrieval import TfidfRetriever, chunk_document, load_chunks
from lakehouse_agent.evals.harness import evaluate
from lakehouse_agent.evals.negative_controls import HallucinatingLLM, OverconfidentLLM, SqlAbuseLLM


def test_chunking_keeps_document_identity():
    chunks = chunk_document("d", "Title\n\n" + "word " * 200 + "\n\n" + "tail " * 200, max_chars=300)
    assert len(chunks) >= 2 and all(c.doc_id == "d" for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]


def test_retrieval_finds_each_policy_doc(env):
    r = TfidfRetriever(load_chunks(env["paths"]["docs"]))
    assert r.search("refund window enterprise")[0]["doc_id"] == "refund_policy"
    assert r.search("how long are invoices retained")[0]["doc_id"] == "data_retention"


def test_orchestration_passes_golden_set_with_offline_llm(env, toolbox):
    """Mechanics check only: RuleBasedLLM is a keyword router, so this does not measure model quality."""
    out = evaluate(Agent(RuleBasedLLM(), toolbox), env["golden"])
    failed = [r["id"] for r in out["rows"] if not r["correct"]]
    assert failed == [], out["rows"]
    s = out["summary"]
    assert s["hallucination_rate"] == 0.0 and s["retrieval_hit_rate"] == 1.0 and s["tool_error_rate"] == 0.0


def test_hybrid_question_uses_both_tools(env, toolbox):
    res = Agent(RuleBasedLLM(), toolbox).ask(
        "What is the SLA first response time for P1 tickets and how many open P1 tickets are there?")
    assert {s.tool for s in res.steps} == {"run_sql", "search_documents"}


def test_harness_flags_fabricated_numbers_and_citations(env, toolbox):
    out = evaluate(Agent(HallucinatingLLM(), toolbox), env["golden"])
    s = out["summary"]
    assert s["hallucination_rate"] >= 0.8
    assert s["answer_accuracy_sql"] < 0.2
    assert any("made_up_policy" in r["ungrounded"]["citations"] for r in out["rows"])


def test_harness_catches_overconfident_answers(env, toolbox):
    out = evaluate(Agent(OverconfidentLLM(), toolbox), env["golden"])
    assert out["summary"]["abstain_accuracy"] == 0.0
    assert out["summary"]["hallucination_rate"] > 0


def test_destructive_and_out_of_scope_sql_is_rejected_not_executed(env, toolbox):
    before = env["spark"].read.parquet(f"{env['s'].root}/gold/customer_360").count()
    res = Agent(SqlAbuseLLM(), toolbox).ask("wipe it")
    assert len(res.steps) == 2 and all("error" in s.result for s in res.steps)
    assert "SqlGuardError" in res.steps[0].result["error"]
    assert env["spark"].read.parquet(f"{env['s'].root}/gold/customer_360").count() == before


def test_step_limit_stops_a_looping_model(env, toolbox):
    class Loop:
        def next(self, messages, tools):
            return {"tool_calls": [{"name": "search_documents", "arguments": {"query": "x"}}]}

    res = Agent(Loop(), toolbox, max_steps=3).ask("anything")
    assert res.hit_step_limit and len(res.steps) == 3


def test_run_is_logged_to_mlflow(env, toolbox, tmp_path):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    evaluate(Agent(RuleBasedLLM(), toolbox), env["golden"], experiment="t", run_name="r", tracking_uri=uri)
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["t"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.hallucination_rate"] == 0.0
    assert runs.iloc[0]["params.llm"] == "RuleBasedLLM"
