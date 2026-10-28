"""Reference-key regression and financial/experience boundary cases."""

import copy
import json
from collections import Counter
from dataclasses import is_dataclass

import pytest
from conftest import COMPANY, KEY, full_requirements

from secop_agent import settings
from secop_agent.eligibility import (
    Check,
    Verdict,
    _compute_financial_indicators,
    validate_company,
    verify_requirements,
)

pytestmark = pytest.mark.usefixtures("offline")


@pytest.mark.parametrize("process", KEY, ids=lambda process: process["file"])
def test_reference_key(process):
    result = verify_requirements(full_requirements(process), COMPANY, process["expected_similar_contracts"])
    assert result["verdict"] == process["expected_verdict"]
    assert all(
        set(check) == {"requirement", "required", "company_value", "passed", "detail"}
        for check in result["checks"]
    )


def test_verdict_counts_and_structured_types():
    counts = Counter(
        verify_requirements(full_requirements(process), COMPANY, process["expected_similar_contracts"])[
            "verdict"
        ]
        for process in KEY
    )
    assert counts == {"ELIGIBLE": 6, "NOT_ELIGIBLE": 2, "NOT_APPLICABLE": 12}
    assert issubclass(Verdict, str) and is_dataclass(Check)


def test_multipurpose_indicators():
    indicators = _compute_financial_indicators(COMPANY, 600_000_000)
    assert indicators == {
        "liquidity_ratio": 2.1,
        "debt_ratio_pct": 55.0,
        "interest_coverage_ratio": 3.2,
        "working_capital": 381_000_000,
        "working_capital_pct_of_budget": 63.5,
        "return_on_equity": 0.156,
        "return_on_assets": 0.07,
        "financial_statements_date": "2025-12-31",
    }


def test_strict_smmlv_changes_process_11():
    requirements = full_requirements(KEY[10])
    assert verify_requirements(requirements, COMPANY, [])["verdict"] == "ELIGIBLE"
    assert verify_requirements(requirements, COMPANY, [], strict=True)["verdict"] == "NOT_ELIGIBLE"


def test_unknown_smmlv_year_is_not_invented():
    requirements = {**full_requirements(KEY[10]), "publication_date": "1990-04-29"}
    assert verify_requirements(requirements, COMPANY, [], True)["verdict"] == "MANUAL_REVIEW"


def test_null_threshold_is_not_missing_evidence():
    requirements = {
        **full_requirements(KEY[17]),
        "experience_by_similar_object": False,
        "experience_max_contracts": None,
        "experience_min_pct_of_budget": None,
    }
    result = verify_requirements(requirements, COMPANY)
    assert result["verdict"] == "ELIGIBLE" and len(result["checks"]) == 2


@pytest.mark.parametrize("budget", [0, None])
def test_incomplete_budget_with_similar_contract(budget):
    requirements = {**full_requirements(KEY[6]), "official_budget": budget}
    result = verify_requirements(requirements, COMPANY, ["EXP-9"])
    assert result["verdict"] == "ELIGIBLE"
    assert any("Incomplete budget" in warning for warning in result["warnings"])
    assert verify_requirements(requirements, COMPANY, [])["verdict"] == "NOT_ELIGIBLE"
    assert verify_requirements(requirements, COMPANY, None)["verdict"] == "MANUAL_REVIEW"


def test_zero_budget_needs_no_strict_conversion():
    company = copy.deepcopy(COMPANY)
    company["experiencia"][-1]["anio"] = 1990
    assert verify_requirements(full_requirements(KEY[6]), company, ["EXP-9"], True)["verdict"] == "ELIGIBLE"


def test_failed_extraction_is_manual_review():
    assert verify_requirements(None, COMPANY)["verdict"] == "MANUAL_REVIEW"


def test_undetermined_evidence_and_failure_priority():
    requirements = {**full_requirements(KEY[10]), "official_budget": None}
    assert verify_requirements(requirements, COMPANY)["verdict"] == "MANUAL_REVIEW"
    requirements["min_liquidity_ratio"] = 5
    assert verify_requirements(requirements, COMPANY)["verdict"] == "NOT_ELIGIBLE"
    requirements.update(
        verifies_financial_capacity=False,
        experience_unspsc_codes=[],
        experience_max_contracts=None,
        experience_min_pct_of_budget=None,
        experience_single_contract_min_pct=None,
    )
    assert verify_requirements(requirements, COMPANY)["verdict"] == "ELIGIBLE"


def test_experience_without_selection_criterion_is_undetermined():
    requirements = {**full_requirements(KEY[10]), "experience_unspsc_codes": []}
    assert verify_requirements(requirements, COMPANY)["verdict"] == "MANUAL_REVIEW"


def test_undetermined_coverage_and_zero_divisors():
    company = copy.deepcopy(COMPANY)
    company["estados_financieros"]["gastos_de_intereses"] = 0
    assert _compute_financial_indicators(company, None)["interest_coverage_ratio"] == "undetermined"
    assert verify_requirements(full_requirements(KEY[10]), company)["verdict"] == "ELIGIBLE"
    company["estados_financieros"]["pasivo_corriente"] = 0
    assert verify_requirements(full_requirements(KEY[10]), company)["verdict"] == "MANUAL_REVIEW"


def test_compare_unrounded_ratios():
    # 727 / 346 is less than 2.105, although displayed as 2.1.
    assert verify_requirements({"min_liquidity_ratio": 2.105}, COMPANY)["verdict"] == "NOT_ELIGIBLE"


def test_select_largest_contracts_and_intersect_codes():
    requirements = {
        "verifies_financial_capacity": False,
        "official_budget": 900_000_000,
        "experience_unspsc_codes": ["V1.81101500", "72141000"],
        "experience_max_contracts": 1,
        "experience_min_pct_of_budget": 100,
    }
    assert verify_requirements(requirements, COMPANY)["verdict"] == "NOT_ELIGIBLE"
    requirements["experience_max_contracts"] = 2
    assert verify_requirements(requirements, COMPANY)["verdict"] == "ELIGIBLE"
    requirements["experience_single_contract_min_pct"] = 70
    assert verify_requirements(requirements, COMPANY)["verdict"] == "NOT_ELIGIBLE"


@pytest.mark.parametrize(
    "company",
    [
        None,
        [],
        {},
        {**COMPANY, "estados_financieros": []},
        {**COMPANY, "experiencia": [1]},
        {**COMPANY, "rup": "yes"},
    ],
)
def test_invalid_profiles(company):
    assert validate_company(company)


def test_invalid_numbers_and_duplicate_contracts():
    company = copy.deepcopy(COMPANY)
    company["estados_financieros"]["activo_total"] = float("nan")
    company["experiencia"].append(company["experiencia"][0])
    assert len(validate_company(company)) == 2


def test_course_company_recorded_verdicts():
    company = json.loads(settings.REFERENCE_COMPANY_FILE.read_text())
    # Similarity decisions belong to this company, not the multipurpose company's key.
    verdicts = {
        process["n"]: verify_requirements(full_requirements(process), company, [])["verdict"]
        for process in KEY
    }
    expected = {number: "NOT_APPLICABLE" for number in (1, 2, 4, 5, 6, 9, 10, 12, 16, 17, 19, 20)}
    expected.update({number: "NOT_ELIGIBLE" for number in (3, 7, 8, 13, 14, 15, 18)})
    expected[11] = "ELIGIBLE"
    assert verdicts == expected
