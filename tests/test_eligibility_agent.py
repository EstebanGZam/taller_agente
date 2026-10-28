"""Offline integration checks for the agent, batch reports, evaluation and Gradio."""

import copy
import csv
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import (
    COMPANY,
    KEY,
    FakeModel,
    FakeStructuredModel,
    full_requirements,
    message,
    similarity_reply,
    tool_call,
)
from langsmith import get_current_run_tree, traceable, tracing_context

from secop_agent import batch_eligibility as batch
from secop_agent import eligibility_agent, settings, tender_documents
from secop_agent import evaluate_eligibility as evaluation
from secop_agent.costs import CostTracker, Usage, cost_usd

pytestmark = pytest.mark.usefixtures("offline")


@pytest.mark.parametrize("process", KEY, ids=lambda process: process["file"])
def test_agent_fallback_all_20(process, fake_extraction):
    agent = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / process["file"], COMPANY)
    result = agent.analyze()
    assert result["verdict"] == process["expected_verdict"]
    assert {"extract_requirements", "verify_requirements"}.issubset(result["tools_used"])
    assert result["metrics"]["model_calls"] == 1
    assert result["verdict"] in result["explanation"]
    assert set(result) == {
        "file",
        "reference",
        "entity",
        "verdict",
        "summary",
        "checks",
        "warnings",
        "explanation",
        "tools_used",
        "metrics",
        "model",
        "run_id",
        "trace_url",
    }
    assert set(result["metrics"]) == {"model_calls", "input_tokens", "output_tokens", "cost_usd", "latency_s"}
    assert agent.analyze() == result
    result["warnings"].append("external mutation")
    assert "external mutation" not in agent.analyze()["warnings"]


def test_memory_and_follow_up_callback(fake_extraction):
    agent = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY)
    agent.analyze()
    state = agent.agent.get_state({"configurable": {"thread_id": agent.thread_id}})
    assert any("Verified result" in str(item.content) for item in state.values["messages"])
    assert agent.ask("¿Por qué?") == "Seguimiento con memoria."
    assert agent.follow_up_metrics[-1]["model_calls"] == 1


def test_callbacks_cover_extraction_and_similarity(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-offline-key")
    agent_model = FakeModel(
        responses=[
            message(tools=[tool_call("extract_requirements")]),
            message(tools=[tool_call("verify_requirements")]),
            message("La empresa cumple los requisitos."),
        ]
    )
    models = iter(
        [
            FakeStructuredModel(responses=[message(json.dumps(full_requirements(KEY[6])))]),
            FakeStructuredModel(responses=[similarity_reply(["EXP-9"], unrelated=["EXP-1"])]),
        ]
    )

    def initialize_agent(name, **kwargs):
        assert not {"temperature", "top_p", "top_k"} & kwargs.keys()
        return agent_model

    def initialize_extractor(name, **kwargs):
        assert not {"temperature", "top_p", "top_k"} & kwargs.keys()
        assert 0 < kwargs["max_tokens"] <= 2400
        return next(models)

    monkeypatch.setattr(eligibility_agent, "chat_model", initialize_agent)
    monkeypatch.setattr(tender_documents, "chat_model", initialize_extractor)
    result = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY).analyze()
    assert result["verdict"] == "ELIGIBLE"
    assert result["metrics"]["model_calls"] == 5
    assert result["metrics"]["input_tokens"] == 500
    assert result["metrics"]["output_tokens"] == 50
    assert result["metrics"]["cost_usd"] == round(5 * cost_usd("claude-haiku-5-5", Usage(100, 10)), 6)
    assert "La empresa" in result["explanation"]


def test_company_and_indicator_tools(fake_extraction):
    agent = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY)
    tools = {item.name: item for item in agent.tools}
    assert set(tools) == {
        "extract_requirements",
        "verify_requirements",
        "get_company_profile",
        "compute_financial_indicators",
    }
    profile = tools["get_company_profile"].invoke({})
    profile["razon_social"] = "mutated"
    assert agent.company == COMPANY
    indicators = tools["compute_financial_indicators"].invoke({"process_budget": 600_000_000})
    assert indicators["working_capital_pct_of_budget"] == 63.5


def test_pdf_read_once(monkeypatch, fake_extraction):
    original, reads = tender_documents.read_pdf, []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(tender_documents, "read_pdf", read)
    agent = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY)
    agent.analyze()
    agent.ask("¿Qué sigue?")
    assert len(reads) == 1


def test_root_trace_metadata_offline(fake_extraction):
    captured = []

    @traceable(name="capture")
    def capture():
        captured.append(get_current_run_tree())
        return eligibility_agent.EligibilityAgent(
            settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY
        ).analyze()

    with tracing_context(enabled="local"):
        result = capture()
    analysis = next(run for run in captured[0].child_runs if run.name == "eligibility_analyze")
    assert result["run_id"] == str(analysis.id)
    assert analysis.metadata["file"] == KEY[6]["file"]
    assert analysis.metadata["model"] == "anthropic:claude-haiku-5-5"


def test_agent_loop_failure_uses_successful_fallback(fake_extraction):
    class FailedAgent:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("Interrupted agent loop")

    agent = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY)
    agent.agent = FailedAgent()
    assert agent.analyze()["verdict"] == "ELIGIBLE"


def test_extraction_failure_is_manual_review(monkeypatch, fake_extraction):
    def fail(*args, **kwargs):
        raise ValueError("Invalid structured output")

    monkeypatch.setattr(tender_documents, "extract_requirements", fail)
    result = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[6]["file"], COMPANY).analyze()
    assert result["verdict"] == "MANUAL_REVIEW"
    assert result["checks"][0]["detail"] == "ValueError"
    assert result["metrics"]["model_calls"] == 1


def test_similarity_failure_does_not_hide_financial_failure(monkeypatch, fake_extraction):
    def extract(*args, **kwargs):
        return {**full_requirements(KEY[17]), "min_liquidity_ratio": 5}

    def fail(*args, **kwargs):
        raise RuntimeError("Similarity unavailable")

    monkeypatch.setattr(tender_documents, "extract_requirements", extract)
    monkeypatch.setattr(tender_documents, "select_similar_contracts", fail)
    result = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[17]["file"], COMPANY).analyze()
    assert result["verdict"] == "NOT_ELIGIBLE"
    assert any(check["passed"] is None for check in result["checks"])
    assert any(check["passed"] is False for check in result["checks"])


def test_parallel_tools_do_not_duplicate_extraction(monkeypatch, fake_extraction):
    calls = []

    def extract(*args, **kwargs):
        calls.append(1)
        return full_requirements(KEY[0])

    monkeypatch.setattr(tender_documents, "extract_requirements", extract)
    agent = eligibility_agent.EligibilityAgent(settings.COURSE_DATA_DIR / KEY[0]["file"], COMPANY)
    config = agent._config(CostTracker(agent.model), "test")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(agent._extract, config), pool.submit(agent._verify, config)]
        for future in futures:
            future.result(timeout=10)
    assert len(calls) == 1
    assert agent.verification["verdict"] == "NOT_APPLICABLE"


def test_dedupe_sha256_and_canonical_name():
    paths = batch.deduplicate_pdfs(settings.COURSE_DATA_DIR.glob("*.pdf"))
    assert len(paths) == 20
    assert "13_LP_DT_CAU_003_2019.pdf" in [path.name for path in paths]
    assert not any("(1)" in path.name for path in paths)


def test_batch_cli_and_report_columns(tmp_path, fake_extraction, monkeypatch, capsys):
    monkeypatch.setenv("TEAM_NAME", "Test team")
    assert batch.main(["--output-dir", str(tmp_path), "--workers", "3"]) == 0
    assert "20/20" in capsys.readouterr().out
    with (tmp_path / "eligibility_processes.csv").open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == batch.CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 20 and rows[10]["official_budget"] == "600000000.0"
    report = json.loads((tmp_path / "eligibility_processes.json").read_text())
    summary = report["summary"]
    assert summary["processes"] == summary["correct"] == 20
    assert summary["accuracy_pct"] == 100
    assert summary["counts"] == {"ELIGIBLE": 6, "NOT_ELIGIBLE": 2, "NOT_APPLICABLE": 12}
    assert summary["team_name"] == "Test team"
    assert summary["input_tokens"] == 2000
    assert summary["total_cost_usd"] > 0 and summary["cost_complete"]
    assert summary["mean_cost_usd"] == summary["total_cost_usd"] / 20
    assert "20/20" in (tmp_path / "eligibility_processes.md").read_text()


def test_key_comparison_excludes_other_companies_and_content(tmp_path, fake_extraction):
    company = copy.deepcopy(COMPANY)
    company["razon_social"] = "Another company"
    report = batch.analyze_batch([settings.COURSE_DATA_DIR / KEY[6]["file"]], company)
    assert report["summary"]["accuracy_pct"] is None
    path = tmp_path / "07_135116.pdf"
    path.write_bytes(b"Different PDF")
    report = batch.analyze_batch([path], COMPANY)
    assert report["rows"][0]["correct"] is None


def test_unreadable_pdf_does_not_abort_batch(tmp_path, monkeypatch, fake_extraction):
    original = tender_documents.read_pdf
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a PDF")

    def read(path):
        if path == broken:
            raise ValueError("Could not read PDF")
        return original(path)

    monkeypatch.setattr(tender_documents, "read_pdf", read)
    report = batch.analyze_batch([broken, settings.COURSE_DATA_DIR / KEY[6]["file"]], COMPANY)
    assert len(report["results"]) == 2
    assert {result["verdict"] for result in report["results"]} == {"MANUAL_REVIEW", "ELIGIBLE"}
    assert not report["summary"]["cost_complete"]


def test_invalid_batch_inputs():
    with pytest.raises(ValueError, match="Worker count"):
        batch.analyze_batch([], COMPANY, workers=0)
    with pytest.raises(ValueError, match="No PDFs"):
        batch.analyze_batch([], COMPANY)
    with pytest.raises(ValueError, match="Invalid company"):
        batch.analyze_batch([], {})


def test_app_import_and_build_without_keys():
    launcher = importlib.import_module("app")
    from secop_agent import web_app

    demo = launcher.build_demo()
    assert demo.config["components"]
    with pytest.raises(web_app.gr.Error, match="ANTHROPIC_API_KEY"):
        web_app.analyze(progress=None)


@pytest.mark.parametrize("colab", [False, True])
def test_launcher_share_setting(monkeypatch, colab):
    import app

    calls = []
    demo = SimpleNamespace(launch=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(app, "build_demo", lambda: SimpleNamespace(queue=lambda: demo))
    if colab:
        monkeypatch.setenv("COLAB_RELEASE_TAG", "")
    else:
        monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    assert app.main([]) == 0
    assert calls == [{"share": colab}]


def test_app_repository_analysis_and_isolated_chat(fake_extraction, monkeypatch, tmp_path):
    from secop_agent import web_app

    monkeypatch.setattr(web_app.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    agents, results, histories, summary, rows, csv_path, choices, detail, conversation = web_app.analyze(
        progress=None
    )
    assert len(rows) == 20 and len(rows[0]) == 9
    assert "20/20" in summary and Path(csv_path).is_file()
    assert "NOT_APPLICABLE" in detail
    _, _, updated = web_app.chat("¿Qué sigue?", "7", agents, histories)
    assert len(updated["7"]) == len(histories["7"]) + 2
    assert updated["1"] == histories["1"]


def test_external_html_is_escaped_and_trace_links_are_safe():
    from secop_agent import web_app

    malicious = '<img src=x onerror="alert(1)">'
    result = {
        "verdict": "ELIGIBLE",
        "reference": malicious,
        "entity": malicious,
        "summary": malicious,
        "explanation": malicious,
        "warnings": [malicious],
        "trace_url": "javascript:alert(1)",
        "run_id": malicious,
        "checks": [
            {
                "requirement": malicious,
                "required": malicious,
                "company_value": malicious,
                "detail": malicious,
                "passed": True,
            }
        ],
    }
    html = web_app.detail_html(result)
    assert "<img" not in html and "&lt;img" in html and "javascript:" not in html
    result["trace_url"] = "https://smith.langchain.com/public/run"
    assert "View LangSmith trace" in web_app.detail_html(result)
    result["trace_url"] = "https://smith.langchain.com.evil.example/run"
    assert "View LangSmith trace" not in web_app.detail_html(result)


def test_evaluators_without_judge():
    expected = {"a": 1.0, "b": ["2", "1"], "c": None, "d": False}
    example = SimpleNamespace(outputs={"verdict": "ELIGIBLE", "requirements": expected})
    run = SimpleNamespace(
        outputs={
            "verdict": "ELIGIBLE",
            "requirements": {"a": 1.0000005, "b": ["1", "2"], "c": None, "d": False},
            "tools_used": ["extract_requirements", "verify_requirements"],
        }
    )
    assert evaluation.verdict_correct(run, example)["score"]
    assert evaluation.requirements_extraction(run, example)["score"] == 1
    assert evaluation.tool_usage(run, example)["score"]
    run.outputs["requirements"]["a"] = 1.01
    del run.outputs["requirements"]["c"]
    assert evaluation.requirements_extraction(run, example)["score"] == 0.5
    run.outputs["tools_used"] = []
    assert not evaluation.tool_usage(run, example)["score"]
    assert not evaluation._matches(1, True)


def test_dataset_created_only_if_missing():
    class FakeClient:
        exists = False
        created = 0
        examples = None

        def has_dataset(self, **kwargs):
            assert kwargs["dataset_name"] == "secop-eligibility-key"
            return self.exists

        def read_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def create_dataset(self, *args, **kwargs):
            self.exists = True
            self.created += 1
            return SimpleNamespace(id="dataset")

        def create_examples(self, *, dataset_id, examples):
            assert dataset_id == "dataset"
            self.examples = examples

    client = FakeClient()
    evaluation.ensure_dataset(client)
    evaluation.ensure_dataset(client)
    assert client.created == 1 and len(client.examples) == 20
    assert set(client.examples[0]["inputs"]) == {"file"}
    assert set(client.examples[0]["outputs"]) == {"verdict", "requirements"}


def test_evaluation_prefix_and_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-offline-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-offline-key")
    monkeypatch.setenv("MODEL", "anthropic:claude-sonnet-5-5")
    captured = []

    class FakeClient:
        def has_dataset(self, **kwargs):
            return True

        def read_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def evaluate(self, target, **kwargs):
            assert target is evaluation.target
            captured.append(kwargs)

    monkeypatch.setattr(evaluation, "Client", FakeClient)
    assert evaluation.main(["eligibility-sonnet"]) == 0
    result = captured[0]
    assert result["experiment_prefix"] == "eligibility-sonnet"
    assert result["data"] == evaluation.DATASET
    assert result["metadata"]["model"] == "anthropic:claude-sonnet-5-5"
    assert len(result["evaluators"]) == 3


def test_evaluation_target_includes_requirements(fake_extraction):
    result = evaluation.target({"file": KEY[6]["file"]})
    assert result["verdict"] == "ELIGIBLE"
    assert result["requirements"]["experience_min_pct_of_budget"] == 80
    with pytest.raises(ValueError, match="directly inside"):
        evaluation.target({"file": "../outside.pdf"})
