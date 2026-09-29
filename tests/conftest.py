import pytest

from lakehouse_agent.agent.retrieval import TfidfRetriever, load_chunks
from lakehouse_agent.agent.tools import ToolBox
from lakehouse_agent.config import Settings
from lakehouse_agent.data.synth import generate
from lakehouse_agent.evals import build_golden
from lakehouse_agent.pipeline.run import run
from lakehouse_agent.spark_session import get_spark


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("lh")
    raw_dir = str(base / "raw")
    paths = generate(raw_dir)
    s = Settings(root=str(base / "lake"))
    spark = get_spark(s, app="tests")
    log = run(spark, {k: paths[k] for k in ("crm", "billing", "support")}, s, batch_id="test")
    golden = build_golden.build(raw_dir)
    return {"spark": spark, "s": s, "raw_dir": raw_dir, "paths": paths, "log": log, "golden": golden}


@pytest.fixture()
def toolbox(env):
    retr = TfidfRetriever(load_chunks(env["paths"]["docs"]))
    return ToolBox(env["spark"], env["s"], retr)
