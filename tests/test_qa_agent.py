"""Q&A agent, evaluator and comparison tests (no keys, no network).

Covers the data behind every reference question as the configuration B tools return it, the
configuration A baseline text, the judge parsing, the tool aliases, the experiment summary, the
comparison report and a fake-model end-to-end run that records tools, tokens and cost.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from secop_agent import compare_experiments, evaluate_qa, qa_agent
from secop_agent import procurement_data as data
from secop_agent.settings import REFERENCE_QUESTIONS_FILE, model_from_env

CASES = {case["id"]: case for case in json.loads(REFERENCE_QUESTIONS_FILE.read_text(encoding="utf-8"))}
TOOLS_A = {t.name: t for t in qa_agent.TOOLS["A"]}
TOOLS_B = {t.name: t for t in qa_agent.TOOLS["B"]}
MODEL = "anthropic:claude-haiku-5-5"


@pytest.fixture(autouse=True)
def sample_mode(monkeypatch):
    monkeypatch.setenv(data.DATA_MODE_ENV, data.SAMPLE_MODE)
    monkeypatch.delenv(qa_agent.AGENT_CONFIG_ENV, raising=False)


def search(**filters):
    return TOOLS_B["search_processes"].invoke(filters)


# --- Configuration A: the course's starter agent ----------------------------------------------

# Copied byte for byte from the course's original agente.py (commit 50c9e33).
ORIGINAL_PROMPT = """Eres un analista de contratación pública colombiana que asesora a una empresa.
Responde en español, de forma breve y precisa.
- Usa las herramientas solo cuando necesites datos; las preguntas generales respóndelas directamente.
- Nunca inventes cifras: si un dato no está o parece inconsistente (por ejemplo, plazo de 0 días
  o presupuesto de $0), dilo explícitamente.
- Cita la referencia del proceso cuando hables de uno."""

ORIGINAL_DESCRIPTIONS = {
    "search_processes": """\
Busca procesos de contratación pública en el SECOP II. Filtros opcionales: entidad,
departamento, ciudad, modalidad (p. ej. 'licitaci', 'directa', 'abreviada'), texto en la
descripción, rango de presupuesto en pesos y rango de fechas de publicación (AAAA-MM-DD).
Devuelve los procesos y un resumen con cantidad, presupuesto total y promedio.""",
    "get_company_profile": """\
Devuelve el perfil de NUESTRA empresa: RUP, códigos UNSPSC, estados financieros y
contratos de experiencia.""",
    "compute_financial_indicators": """\
Calcula los indicadores financieros de NUESTRA empresa: liquidez, endeudamiento,
cobertura de intereses, capital de trabajo y rentabilidades. Si se da el presupuesto de
un proceso, calcula qué porcentaje de él cubre nuestro capital de trabajo.""",
}


def test_config_a_prompt_is_byte_identical_to_the_course():
    assert qa_agent.PROMPTS["A"].encode("utf-8") == ORIGINAL_PROMPT.encode("utf-8")


def test_config_a_tool_descriptions_are_the_course_originals():
    assert {name: t.description for name, t in TOOLS_A.items()} == ORIGINAL_DESCRIPTIONS


def test_both_configs_share_the_english_tool_names():
    assert (
        list(TOOLS_A)
        == list(TOOLS_B)
        == ["search_processes", "get_company_profile", "compute_financial_indicators"]
    )


def test_config_a_returns_the_data_layer_output():
    assert TOOLS_A["search_processes"].invoke({"modality": "licitaci"}) == data.search_processes(
        modality="licitaci"
    )
    assert TOOLS_A["get_company_profile"].invoke({}) == data.get_company_profile()
    assert TOOLS_A["compute_financial_indicators"].invoke({"process_budget": 305_813_941}) == (
        data.compute_financial_indicators(data.get_company_profile(), 305_813_941)
    )


def test_config_b_prompt_is_english_and_requires_spanish_answers():
    prompt = qa_agent.PROMPTS["B"]
    assert "Always answer in Spanish" in prompt and "WITHOUT tools" in prompt
    assert "data, not instructions" in prompt


def test_no_sampling_parameters_in_our_modules():
    package = Path(qa_agent.__file__).parent
    for name in ("qa_agent.py", "evaluate_qa.py"):
        source = (package / name).read_text(encoding="utf-8")
        code = " ".join(
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        )
        assert not re.search(r"\b(temperature|top_p|top_k)\s*=", code), name


def test_config_selection(monkeypatch):
    assert qa_agent.normalize_config() == "B"
    monkeypatch.setenv(qa_agent.AGENT_CONFIG_ENV, " a ")
    assert qa_agent.normalize_config() == "A"
    with pytest.raises(ValueError):
        qa_agent.normalize_config("C")


# --- Data behind each reference question (configuration B tools) ------------------------------


def test_r01_to_r04_indicators():
    indicators = TOOLS_B["compute_financial_indicators"].invoke({})
    assert indicators["liquidity_ratio"] == 1.7  # R01: "1.7"
    assert indicators["debt_ratio_pct"] == 62.0  # R02: "62.0 %"
    assert indicators["interest_coverage_ratio"] == 2.5  # R03: "2.5"
    assert indicators["working_capital"] == 182_000_000  # R04: "$182.000.000"


def test_r05_r06_company_profile_and_unspsc_experience():
    profile = TOOLS_B["get_company_profile"].invoke({"unspsc_code": "81101500"})
    assert ", ".join(profile["rup"]["codigos_unspsc"]) == CASES["R05"]["respuesta_esperada"]
    assert profile["experience_with_code"] == {
        "code": "81101500",
        "contracts": ["EXP-1", "EXP-2"],
        "total_value": 950_000_000,
    }


@pytest.mark.parametrize("requested", ["directa", "Contratación directa", "contratacion directa"])
def test_r07_nine_direct_contracting_processes(requested):
    result = search(modality=requested)
    assert result["summary"]["count"] == 9 == int(CASES["R07"]["respuesta_esperada"])
    assert all(p["modality"] == "Contratación directa" for p in result["processes"])


def test_r07_direct_does_not_count_other_modalities():
    assert not qa_agent.modality_matches("directa", "Contratación directa (con ofertas)")
    assert not qa_agent.modality_matches("directa", "Contratación régimen especial")
    assert qa_agent.modality_matches("régimen especial", "Contratación régimen especial (con ofertas)")


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_r08_largest_budget_survives_small_limits(limit):
    result = search(limit=limit)
    assert result["summary"]["count"] == 20 and len(result["processes"]) == limit
    assert result["summary"]["largest_budget"] == {
        "reference": "LP-DT-CAU-003-2019",
        "entity": "INVIAS",
        "official_budget": 53_520_784_298,
    }
    assert search(sort_by="budget", limit=1)["processes"][0]["reference"] == "LP-DT-CAU-003-2019"


def test_r09_public_tenders():
    assert search(modality="licitación pública")["summary"]["references"] == [
        "FDLS-LP-098-2019",
        "LP-DT-CAU-003-2019",
    ]


def test_r10_antioquia_average_budget():
    summary = search(department="Antioquia")["summary"]
    assert summary["count"] == 3 and round(summary["average_budget"]) == 120_120_571


def test_r11_santander_most_recent():
    result = search(department="Santander")
    assert result["summary"]["count"] == 2
    assert result["summary"]["most_recent"] == {
        "reference": "FLO-INFRA-SA-MC-007-2025",
        "publication_date": "2025-12-30",
    }
    assert (
        search(department="Santander", sort_by="date")["processes"][0]["reference"]
        == "FLO-INFRA-SA-MC-007-2025"
    )


def test_r12_contracting_unit_city():
    process = next(
        p
        for p in search(entity="INVIAS")["processes"]
        if p["contracting_unit"] == "Dirección Territorial Cauca"
    )
    assert (process["reference"], process["entity_city"], process["contracting_unit_city"]) == (
        "LP-DT-CAU-003-2019",
        "Bogotá",
        "Cali",
    )


def test_r13_bids_and_awarded_supplier():
    [process] = search(reference="lp-dt-cau-003-2019")["processes"]
    assert process["bids_received"] == 51
    assert process["awarded_supplier"] == "COOPERATIVA DE TRABAJO ASOCIADO SIGLO XXI"
    assert process["awarded_value"] == 1_217_527_123


def test_r14_zero_duration_is_flagged():
    [process] = search(reference="SA-004-2023")["processes"]
    assert process["duration"].startswith("0 ") and qa_agent.ZERO_DURATION_WARNING in process["warnings"]


def test_r15_zero_budget_is_flagged():
    [process] = search(entity="Agrosavia")["processes"]
    assert process["official_budget"] == 0 and qa_agent.ZERO_BUDGET_WARNING in process["warnings"]


def test_r16_working_capital_against_40_percent():
    [process] = search(reference="FLO-INFRA-SA-MC-007-2025")["processes"]
    assert process["official_budget"] == 305_813_941 and "warnings" not in process
    indicators = TOOLS_B["compute_financial_indicators"].invoke(
        {"process_budget": process["official_budget"], "required_pct": 40}
    )
    assert indicators["working_capital_pct_of_budget"] == 59.5
    assert indicators["required_amount"] == 122_325_576
    assert indicators["meets_working_capital_requirement"] is True


def test_required_pct_without_budget_gives_a_notice():
    indicators = TOOLS_B["compute_financial_indicators"].invoke({"process_budget": 0, "required_pct": 40})
    assert "notice" in indicators and "required_amount" not in indicators


def test_r17_r18_expect_no_tools():
    assert CASES["R17"]["herramientas_esperadas"] == [] == CASES["R18"]["herramientas_esperadas"]


def test_reference_lookup_falls_back_to_secop_id():
    [process] = search(reference="CO1.REQ.2577563")["processes"]
    assert process["reference"] == "EDP-545-2022"


def test_config_b_payload_is_smaller_than_a():
    size_b = len(json.dumps(search(limit=20), ensure_ascii=False))
    size_a = len(json.dumps(TOOLS_A["search_processes"].invoke({}), ensure_ascii=False))
    assert size_b < size_a


# --- Message helpers ---------------------------------------------------------------------------


def test_tools_used_with_dicts_and_objects():
    messages = [
        SimpleNamespace(tool_calls=[{"name": "search_processes", "args": {}}]),
        SimpleNamespace(tool_calls=[SimpleNamespace(name="compute_financial_indicators")]),
        SimpleNamespace(content="no tools"),
    ]
    assert qa_agent.tools_used(messages) == ["search_processes", "compute_financial_indicators"]


def test_message_text_from_blocks():
    assert qa_agent.message_text("hola") == "hola"
    blocks = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "1.7"}]
    assert qa_agent.message_text(blocks) == "1.7"


# --- Evaluator ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict, expected",
    [
        (evaluate_qa.JudgeVerdict(reason="ok", correct=True), 1),
        (evaluate_qa.JudgeVerdict(reason="no", correct=False), 0),
        (
            {
                "raw": AIMessage(content="x"),
                "parsed": evaluate_qa.JudgeVerdict(reason="ok", correct=True),
                "parsing_error": None,
            },
            1,
        ),
        ({"raw": AIMessage(content="INCORRECT: wrong figure"), "parsed": None, "parsing_error": "bad"}, 0),
        ({"correct": False, "reason": "no"}, 0),
        ("CORRECT", 1),
        ("INCORRECT", 0),
        ("incorrect.", 0),
        ("Correct", 1),
        ("**INCORRECTA**: no", 0),
        ("CORRECTA", 1),
        ([{"type": "text", "text": "CORRECT"}], 1),
        ("not sure", None),
        ("INCORRECTLY", None),
    ],
)
def test_parse_verdict(verdict, expected):
    assert evaluate_qa.parse_verdict(verdict) == expected


def test_reference_tool_aliases():
    assert evaluate_qa.REFERENCE_TOOL_ALIASES == {
        "buscar_procesos": "search_processes",
        "consultar_empresa": "get_company_profile",
        "calcular_indicadores": "compute_financial_indicators",
    }
    expected = {name for case in CASES.values() for name in case["herramientas_esperadas"]}
    assert evaluate_qa.canonical_tool_names(sorted(expected)) == set(TOOLS_B)


def test_tool_usage_maps_spanish_reference_names():
    reference = {"herramientas": ["buscar_procesos", "calcular_indicadores"]}

    def score(used, ref=reference):
        return evaluate_qa.tool_usage({}, {"tools_used": used}, ref)["score"]

    assert score(["search_processes", "compute_financial_indicators"]) == 1
    assert score(["search_processes", "compute_financial_indicators", "get_company_profile"]) == 1
    assert score(["search_processes"]) == 0
    assert score([], {"herramientas": []}) == 1
    assert score(["get_company_profile"], {"herramientas": []}) == 0


def test_correct_answer_uses_the_structured_judge(monkeypatch):
    prompts = []

    class FakeJudge:
        def invoke(self, prompt):
            prompts.append(prompt)
            return {
                "raw": AIMessage(content=""),
                "parsing_error": None,
                "parsed": evaluate_qa.JudgeVerdict(reason="Same figure.", correct=True),
            }

    monkeypatch.setattr(evaluate_qa, "_judge", FakeJudge())
    result = evaluate_qa.correct_answer({"pregunta": "¿Liquidez?"}, {"answer": "1,7"}, {"respuesta": "1.7"})
    assert result == {"key": "correct_answer", "score": 1, "comment": "Same figure."}
    assert "EXPECTED ANSWER: 1.7" in prompts[0] and "ANSWER: 1,7" in prompts[0]


def test_percentile_interpolates():
    values = [float(i) for i in range(1, 19)]
    assert evaluate_qa.percentile(values, 50) == 9.5
    assert evaluate_qa.percentile(values, 95) == pytest.approx(17.15)
    assert evaluate_qa.percentile([], 50) is None and evaluate_qa.percentile([3.0], 95) == 3.0


def make_rows(n, correct, cost=0.001):
    return [
        {
            "id": f"R{i + 1:02d}",
            "question": "",
            "tools_used": [],
            "answer": "",
            "tool_usage": 1,
            "correct_answer": int(i < correct),
            "input_tokens": 1000,
            "output_tokens": 100,
            "cost_usd": cost,
            "latency_s": float(i + 1),
            "error": None,
        }
        for i in range(n)
    ]


def test_summarize_experiment():
    summary = evaluate_qa.summarize_experiment(
        make_rows(18, 16), "B-improved", "B", MODEL, langsmith_cost=0.02
    )
    assert summary["answer_accuracy"] == 88.9 and summary["tool_usage_accuracy"] == 100.0
    assert summary["combined_accuracy"] == 88.9
    assert summary["input_tokens"] == 18_000 and summary["total_cost_usd"] == 0.018
    assert summary["cost_per_question_usd"] == 0.001
    assert summary["cost_per_correct_answer_usd"] == round(0.018 / 16, 6)
    assert summary["langsmith_cost_usd"] == 0.02
    assert summary["latency_p50_s"] == 9.5 and summary["latency_p95_s"] == 17.15
    assert summary["agent_config"] == "B" and summary["change"] == qa_agent.CONFIG_DESCRIPTIONS["B"]


def test_save_summary_path(tmp_path):
    summary = evaluate_qa.summarize_experiment(make_rows(2, 2), "A-baseline", "A", MODEL)
    path = evaluate_qa.save_summary(summary, tmp_path)
    assert path == tmp_path / "qa_experiment_A-baseline.json"
    assert json.loads(path.read_text(encoding="utf-8"))["answer_accuracy"] == 100.0


def test_row_from_langsmith_result():
    from langsmith.evaluation import EvaluationResult

    row = evaluate_qa.row_from_result(
        {
            "run": SimpleNamespace(
                outputs={
                    "answer": "9",
                    "tools_used": ["search_processes"],
                    "metrics": {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.1, "latency_s": 1.5},
                },
                error=None,
                start_time=None,
                end_time=None,
            ),
            "example": SimpleNamespace(inputs={"pregunta": CASES["R07"]["pregunta"]}),
            "evaluation_results": {
                "results": [
                    EvaluationResult(key="tool_usage", score=1),
                    EvaluationResult(key="correct_answer", score=0),
                ]
            },
        },
        {CASES["R07"]["pregunta"]: "R07"},
    )
    assert row["id"] == "R07" and row["tool_usage"] == 1 and row["correct_answer"] == 0
    assert row["cost_usd"] == 0.1 and row["latency_s"] == 1.5 and row["tools_used"] == ["search_processes"]


def test_evaluate_main_requires_keys(monkeypatch, capsys):
    monkeypatch.setattr(evaluate_qa, "load_environment", lambda: None)
    for name in evaluate_qa.REQUIRED_KEYS:
        monkeypatch.delenv(name, raising=False)
    assert evaluate_qa.main(["X"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# --- Comparison report -------------------------------------------------------------------------


def write_experiments(results_dir):
    for prefix, config, correct, cost in (("A-baseline", "A", 12, 0.002), ("B-improved", "B", 17, 0.001)):
        summary = evaluate_qa.summarize_experiment(make_rows(18, correct, cost), prefix, config, MODEL)
        evaluate_qa.save_summary(summary, results_dir)


def test_comparison_report(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    write_experiments(tmp_path)
    experiments = [compare_experiments.load_experiment(p, tmp_path) for p in ("A-baseline", "B-improved")]
    report = compare_experiments.build_report(experiments)
    assert "| | A-baseline | B-improved |" in report
    assert "| Correct answer | 66.7 % | 94.4 % |" in report
    assert "| Configuration | A | B |" in report
    assert "Meeting the 85% correct-answer criterion: B-improved (94.4%)." in report
    assert "Cheapest per correct answer: B-improved (USD 0.001059)." in report
    assert "Recommended: B-improved" in report


def test_comparison_verdict_when_nothing_passes():
    experiments = [evaluate_qa.summarize_experiment(make_rows(18, 10), "A-baseline", "A", MODEL)]
    assert "criterion: none." in compare_experiments.verdict(experiments)
    assert "Recommended" not in compare_experiments.verdict(experiments)


def test_comparison_cli_writes_markdown(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(compare_experiments, "load_environment", lambda: None)
    write_experiments(tmp_path)
    assert (
        compare_experiments.main(["A-baseline", "B-improved", "missing", "--results-dir", str(tmp_path)]) == 0
    )
    assert "| | A-baseline | B-improved |" in (tmp_path / "qa_comparison.md").read_text(encoding="utf-8")
    assert "missing" in capsys.readouterr().err
    assert compare_experiments.main(["missing", "--results-dir", str(tmp_path)]) == 1


# --- End to end with a fake model --------------------------------------------------------------


class ScriptedChatModel(GenericFakeChatModel):
    """Chat model that replays fixed messages (no network) and accepts tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted_agent(config, messages):
    model = ScriptedChatModel(messages=iter(messages))
    return create_agent(model, tools=qa_agent.TOOLS[config], system_prompt=qa_agent.PROMPTS[config])


def test_import_and_ask_need_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(
        qa_agent._AGENT_CACHE, ("A", MODEL), scripted_agent("A", [AIMessage(content="Hola.")])
    )
    result = qa_agent.ask("¿Qué es el SECOP II?", config="A", model=MODEL)
    assert result == qa_agent.QAResult(answer="Hola.", tools_used=[])


def test_fake_model_run_records_tools_tokens_and_cost(monkeypatch):
    usage = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}
    metadata = {"model_name": "claude-haiku-5-5"}
    agent = scripted_agent(
        "B",
        [
            AIMessage(
                content=[{"type": "text", "text": "Busco."}],
                usage_metadata=usage,
                response_metadata=metadata,
                tool_calls=[{"name": "search_processes", "args": {"modality": "directa"}, "id": "t1"}],
            ),
            AIMessage(
                content=[{"type": "text", "text": "Hay 9."}], usage_metadata=usage, response_metadata=metadata
            ),
        ],
    )
    monkeypatch.setitem(qa_agent._AGENT_CACHE, ("B", model_from_env()), agent)
    outputs = evaluate_qa.run_agent({"pregunta": CASES["R07"]["pregunta"]})
    assert outputs["answer"] == "Hay 9." and outputs["tools_used"] == ["search_processes"]
    metrics = outputs["metrics"]
    assert (metrics["model_calls"], metrics["input_tokens"], metrics["output_tokens"]) == (2, 2000, 100)
    assert metrics["cost_usd"] == pytest.approx(2000 * 0.10 / 1e6 + 100 * 0.50 / 1e6)
    assert metrics["cost_complete"] is True and metrics["latency_s"] >= 0
    reference = {"herramientas": CASES["R07"]["herramientas_esperadas"]}
    assert evaluate_qa.tool_usage({}, outputs, reference)["score"] == 1
    assert evaluate_qa.cost_usd(outputs)["score"] == metrics["cost_usd"]
