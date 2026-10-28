"""Data layer tests (no keys, no network): sample search, SoQL building and RUP indicators."""

from __future__ import annotations

import pytest
import requests

from secop_agent import procurement_data as data
from secop_agent.settings import MULTIPURPOSE_COMPANY_FILE, REFERENCE_COMPANY_FILE


@pytest.fixture(autouse=True)
def sample_mode(monkeypatch):
    monkeypatch.setenv(data.DATA_MODE_ENV, data.SAMPLE_MODE)


def course_indicators(statements: dict, process_budget: float | None = None) -> dict:
    """The course's `herramientas.calcular_indicadores` formulas, written out independently."""
    ef = statements
    expected = {
        "liquidity_ratio": round(ef["activo_corriente"] / ef["pasivo_corriente"], 2),
        "debt_ratio_pct": round(100 * ef["pasivo_total"] / ef["activo_total"], 1),
        "interest_coverage_ratio": (
            round(ef["utilidad_operacional"] / ef["gastos_de_intereses"], 2)
            if ef["gastos_de_intereses"]
            else "undetermined"
        ),
        "working_capital": ef["activo_corriente"] - ef["pasivo_corriente"],
        "return_on_equity": round(ef["utilidad_operacional"] / ef["patrimonio"], 3),
        "return_on_assets": round(ef["utilidad_operacional"] / ef["activo_total"], 3),
        "financial_statements_date": ef["corte"],
    }
    if process_budget:
        expected["working_capital_pct_of_budget"] = round(
            100 * (ef["activo_corriente"] - ef["pasivo_corriente"]) / float(process_budget), 1
        )
    return expected


# --- Financial indicators ----------------------------------------------------------------------


def test_reference_company_indicators_r01_to_r04():
    indicators = data.compute_financial_indicators(data.get_company_profile())
    assert indicators["liquidity_ratio"] == 1.7  # R01
    assert indicators["debt_ratio_pct"] == 62.0  # R02
    assert indicators["interest_coverage_ratio"] == 2.5  # R03
    assert indicators["working_capital"] == 182_000_000  # R04
    assert "working_capital_pct_of_budget" not in indicators


@pytest.mark.parametrize("company_file", [REFERENCE_COMPANY_FILE, MULTIPURPOSE_COMPANY_FILE])
@pytest.mark.parametrize("budget", [None, 0, 305_813_941, 1_000_000.0])
def test_indicator_parity_with_course_formulas(company_file, budget):
    company = data.load_company(company_file)
    assert data.compute_financial_indicators(company, budget) == course_indicators(
        company["estados_financieros"], budget
    )


def test_r16_working_capital_share_of_budget():
    indicators = data.compute_financial_indicators(data.get_company_profile(), 305_813_941)
    assert indicators["working_capital_pct_of_budget"] == 59.5


def test_zero_interest_is_undetermined():
    company = data.get_company_profile()
    company["estados_financieros"]["gastos_de_intereses"] = 0
    assert data.compute_financial_indicators(company)["interest_coverage_ratio"] == data.UNDETERMINED


def test_spec_indicator_keys():
    keys = set(data.compute_financial_indicators(data.get_company_profile(), 1))
    assert keys == {
        "liquidity_ratio",
        "debt_ratio_pct",
        "interest_coverage_ratio",
        "working_capital",
        "return_on_equity",
        "return_on_assets",
        "financial_statements_date",
        "working_capital_pct_of_budget",
    }


def test_r05_r06_company_profile():
    profile = data.get_company_profile()
    assert profile["rup"]["codigos_unspsc"] == ["72141000", "81101500", "72101500", "72152700", "72102900"]
    experience = [e for e in profile["experiencia"] if "81101500" in e["codigos_unspsc"]]
    assert [e["id"] for e in experience] == ["EXP-1", "EXP-2"]
    assert sum(e["valor"] for e in experience) == 950_000_000


# --- Sample search -----------------------------------------------------------------------------


def test_search_without_filters_returns_the_whole_sample():
    result = data.search_processes(limit=50)
    assert result["summary"]["count"] == 20 and len(result["processes"]) == 20
    assert result["summary"]["zero_budget_processes"] == 1


def test_default_limit_and_clamping():
    assert len(data.search_processes()["processes"]) == data.DEFAULT_LIMIT
    assert len(data.search_processes(limit=0)["processes"]) == data.DEFAULT_LIMIT
    assert len(data.search_processes(limit=-5)["processes"]) == 1
    assert data.clamp_limit(500) == data.MAX_LIMIT


def test_r07_direct_contracting_count():
    assert data.search_processes(modality="directa")["summary"]["count"] == 9


def test_r08_largest_budget_in_sample():
    processes = data.search_processes(limit=50)["processes"]
    largest = max(processes, key=lambda p: data.to_number(p["precio_base"]))
    assert (largest["referencia_del_proceso"], largest["entidad"], int(largest["precio_base"])) == (
        "LP-DT-CAU-003-2019",
        "INVIAS",
        53_520_784_298,
    )


def test_r09_public_tenders():
    result = data.search_processes(modality="licitación pública")
    assert [p["referencia_del_proceso"] for p in result["processes"]] == [
        "FDLS-LP-098-2019",
        "LP-DT-CAU-003-2019",
    ]


def test_r10_antioquia_average_budget():
    summary = data.search_processes(department="Antioquia")["summary"]
    assert summary["count"] == 3 and round(summary["average_budget"]) == 120_120_571


def test_r11_santander_most_recent():
    processes = data.search_processes(department="santander")["processes"]
    latest = max(processes, key=data.publication_date)
    assert len(processes) == 2
    assert (latest["referencia_del_proceso"], data.publication_date(latest)) == (
        "FLO-INFRA-SA-MC-007-2025",
        "2025-12-30",
    )


def test_city_filter_matches_contracting_unit_city():
    references = [p["referencia_del_proceso"] for p in data.search_processes(city="Cali")["processes"]]
    assert "LP-DT-CAU-003-2019" in references  # INVIAS is registered in Bogotá; its unit is in Cali.


def test_budget_and_date_filters():
    result = data.search_processes(
        min_budget=300_000_000, max_budget=1_000_000_000, date_from="2019-01-01", date_to="2024-12-31"
    )
    assert {p["referencia_del_proceso"] for p in result["processes"]} == {
        "FDLS-LP-098-2019",
        "FNV-001-2021",
        "0511-2024",
    }


def test_text_filter_is_accent_insensitive():
    assert data.normalize_text("Contratación DIRECTA") == "contratacion directa"
    assert data.search_processes(text="migracion")["summary"]["count"] == 1


def test_summary_of_nothing():
    assert data.summarize([]) == {
        "count": 0,
        "total_budget": 0,
        "average_budget": 0,
        "zero_budget_processes": 0,
    }


# --- Live API mode (mocked) --------------------------------------------------------------------


def test_soql_where_escapes_quotes():
    where = data.build_soql_where(
        data.ProcessFilters(entity="O'Higgins", min_budget=10, date_to="2025-01-31")
    )
    assert "upper(entidad) like '%O''HIGGINS%'" in where
    assert "precio_base >= 10.0" in where and "fecha_de_publicacion_del <= '2025-01-31T23:59:59'" in where
    assert data.build_soql_where(data.ProcessFilters()) is None


def test_api_mode_queries_datos_gov_co(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"referencia_del_proceso": "X-1", "precio_base": "100"}]

    def fake_get(url, params, headers, timeout):
        calls.update(url=url, params=params, headers=headers)
        return FakeResponse()

    monkeypatch.setenv(data.DATA_MODE_ENV, data.API_MODE)
    monkeypatch.setenv("SOCRATA_APP_TOKEN", "token")
    monkeypatch.setattr(data.requests, "get", fake_get)
    result = data.search_processes(department="Valle", limit=5)
    assert result["summary"]["count"] == 1 and result["summary"]["total_budget"] == 100
    assert calls["url"] == data.API_URL and calls["params"]["$limit"] == 5
    assert calls["params"]["$where"] == "upper(departamento_entidad) like '%VALLE%'"
    assert calls["headers"] == {"X-App-Token": "token"}


def test_api_errors_are_returned_not_raised(monkeypatch):
    def failing_get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setenv(data.DATA_MODE_ENV, data.API_MODE)
    monkeypatch.setattr(data.requests, "get", failing_get)
    assert "offline" in data.search_processes()["error"]


def test_smoke_test_cli(capsys):
    assert data.main([]) == 0
    output = capsys.readouterr().out
    assert "'liquidity_ratio': 1.7" in output and "'count': 2" in output


def test_sample_file_is_the_course_data():
    processes = data.load_sample_processes()
    assert len(processes) == 20 and len({p["referencia_del_proceso"] for p in processes}) == 20
