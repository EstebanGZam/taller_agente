"""Deterministic financial and experience checks against explicit tender requirements."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

# Course reference table in COP, not a live source of wage data.
SMMLV = {
    2018: 781_242,
    2019: 828_116,
    2020: 877_803,
    2021: 908_526,
    2022: 1_000_000,
    2023: 1_160_000,
    2024: 1_300_000,
    2025: 1_423_500,
    2026: 1_750_905,
}
FINANCIAL_FIELDS = (
    "activo_corriente",
    "pasivo_corriente",
    "activo_total",
    "pasivo_total",
    "patrimonio",
    "utilidad_operacional",
    "gastos_de_intereses",
)


class Verdict(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True)
class Check:
    requirement: str
    required: Any
    company_value: Any
    passed: bool | None
    detail: str = ""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_company(company: dict) -> list[str]:
    """Return profile errors; preserve the Spanish keys of the course data contract."""
    if not isinstance(company, dict):
        return ["Company JSON must be an object."]
    errors = []
    if not isinstance(company.get("razon_social"), str) or not company["razon_social"].strip():
        errors.append("A valid company name (razon_social) is required.")
    registration = company.get("rup")
    if not isinstance(registration, dict) or not isinstance(registration.get("vigente"), bool):
        errors.append("rup.vigente must be a boolean.")
    elif not isinstance(registration.get("codigos_unspsc"), list):
        errors.append("rup.codigos_unspsc must be a list.")
    financials = company.get("estados_financieros")
    if not isinstance(financials, dict):
        errors.append("estados_financieros must be an object.")
    else:
        for field in FINANCIAL_FIELDS:
            if not _is_number(financials.get(field)):
                errors.append(f"estados_financieros.{field} must be a finite number.")
            elif field not in {"patrimonio", "utilidad_operacional"} and financials[field] < 0:
                errors.append(f"estados_financieros.{field} cannot be negative.")
    experience = company.get("experiencia")
    if not isinstance(experience, list):
        errors.append("experiencia must be a list.")
    else:
        identifiers = set()
        for index, contract in enumerate(experience, 1):
            if not isinstance(contract, dict):
                errors.append(f"Contract {index} must be an object.")
                continue
            identifier = contract.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                errors.append(f"Contract {index} requires a unique string id.")
            else:
                identifiers.add(identifier)
            if not isinstance(contract.get("objeto"), str) or not contract["objeto"].strip():
                errors.append(f"Contract {index} requires an object description (objeto).")
            if not _is_number(contract.get("valor")) or contract["valor"] < 0:
                errors.append(f"Contract {index} requires a nonnegative value (valor).")
            if not isinstance(contract.get("codigos_unspsc"), list):
                errors.append(f"Contract {index} UNSPSC codes must be a list.")
            if not isinstance(contract.get("anio"), int) or isinstance(contract.get("anio"), bool):
                errors.append(f"Contract {index} year (anio) must be an integer.")
    return errors


def _raw_financial_indicators(company: dict, process_budget: float | None) -> dict:
    financials = company["estados_financieros"]

    def ratio(numerator: str, denominator: str) -> float | None:
        return financials[numerator] / financials[denominator] if financials[denominator] else None

    indicators = {
        "liquidity_ratio": ratio("activo_corriente", "pasivo_corriente"),
        "debt_ratio_pct": (
            100 * financials["pasivo_total"] / financials["activo_total"]
            if financials["activo_total"]
            else None
        ),
        "interest_coverage_ratio": (
            ratio("utilidad_operacional", "gastos_de_intereses")
            if financials["gastos_de_intereses"]
            else "undetermined"
        ),
        "working_capital": financials["activo_corriente"] - financials["pasivo_corriente"],
        "return_on_equity": ratio("utilidad_operacional", "patrimonio"),
        "return_on_assets": ratio("utilidad_operacional", "activo_total"),
        "financial_statements_date": financials.get("corte"),
    }
    if process_budget:
        indicators["working_capital_pct_of_budget"] = 100 * indicators["working_capital"] / process_budget
    return indicators


# TODO(merge): use procurement_data.compute_financial_indicators
# Keep raw ratios above for decisions; the shared helper rounds only for reporting.
def _compute_financial_indicators(company: dict, process_budget: float | None) -> dict:
    """Match the shared RUP indicator keys and the course's display rounding."""
    indicators = _raw_financial_indicators(company, process_budget)
    decimals = {
        "liquidity_ratio": 2,
        "debt_ratio_pct": 1,
        "interest_coverage_ratio": 2,
        "return_on_equity": 3,
        "return_on_assets": 3,
        "working_capital_pct_of_budget": 1,
    }
    return {
        key: round(value, decimals[key]) if key in decimals and _is_number(value) else value
        for key, value in indicators.items()
    }


def _financial_checks(requirements: dict, indicators: dict) -> list[Check]:
    if requirements.get("verifies_financial_capacity") is False:
        return [Check("Financial capacity", "Not assessed", "Not required", True)]
    checks = []
    thresholds = [
        ("Liquidity", "min_liquidity_ratio", "liquidity_ratio", False),
        ("Debt (%)", "max_debt_ratio_pct", "debt_ratio_pct", True),
        ("Interest coverage", "min_interest_coverage", "interest_coverage_ratio", False),
        ("Return on equity", "min_return_on_equity", "return_on_equity", False),
        ("Return on assets", "min_return_on_assets", "return_on_assets", False),
    ]
    for name, field, indicator, maximum in thresholds:
        threshold = requirements.get(field)
        if threshold is None:
            continue
        value = indicators[indicator]
        passed = (
            True
            if value == "undetermined"
            else None
            if value is None
            else value <= threshold
            if maximum
            else value >= threshold
        )
        checks.append(Check(name, f"{'≤' if maximum else '≥'} {threshold}", value, passed))
    percentage = requirements.get("min_working_capital_pct_of_budget")
    if percentage is not None:
        budget = requirements.get("official_budget")
        minimum = budget * percentage / 100 if budget else None
        value = indicators["working_capital"]
        checks.append(
            Check(
                "Working capital",
                f"≥ {percentage}% of budget",
                value,
                None if minimum is None else value >= minimum,
                "A valid budget is missing." if minimum is None else f"Minimum: {minimum:,.2f} COP",
            )
        )
    return checks


def _experience_amount_checks(requirements: dict, contracts: list[dict], strict: bool) -> list[Check]:
    if not contracts:
        return [Check("Experience", "At least one qualifying contract", "None", False)]
    budget = requirements.get("official_budget") or 0
    convert = strict and bool(budget)
    publication_wage = 1
    if convert:
        try:
            publication_wage = SMMLV[int(str(requirements.get("publication_date"))[:4])]
        except (ValueError, KeyError):
            publication_wage = None
    values = [
        (
            contract,
            contract["valor"] / SMMLV[contract["anio"]]
            if convert and contract["anio"] in SMMLV
            else contract["valor"]
            if not convert
            else None,
        )
        for contract in contracts
    ]
    if publication_wage is None or any(value is None for _, value in values):
        return [Check("Experience in SMMLV", "Years present in the reference table", "Unknown year", None)]
    values.sort(key=lambda pair: pair[1], reverse=True)
    selected = values[: int(requirements["experience_max_contracts"])]
    total = sum(value for _, value in selected)
    minimum = budget / publication_wage * requirements["experience_min_pct_of_budget"] / 100
    unit = "SMMLV" if convert else "COP"
    checks = [
        Check(
            "Combined experience",
            f"≥ {minimum:,.2f} {unit}",
            total,
            total >= minimum,
            "Contracts: " + ", ".join(contract["id"] for contract, _ in selected),
        )
    ]
    single_percentage = requirements.get("experience_single_contract_min_pct")
    if single_percentage is not None:
        minimum = budget / publication_wage * single_percentage / 100
        checks.append(
            Check(
                "Single-contract experience",
                f"≥ {minimum:,.2f} {unit}",
                values[0][1],
                values[0][1] >= minimum,
            )
        )
    return checks


def _experience_checks(
    requirements: dict,
    company: dict,
    similar_contract_ids: list[str] | None,
    strict: bool,
    warnings: list[str],
) -> list[Check]:
    codes = {str(code).removeprefix("V1.") for code in requirements.get("experience_unspsc_codes", [])}
    by_object = bool(requirements.get("experience_by_similar_object")) and not codes
    maximum, percentage, single_percentage = (
        requirements.get("experience_max_contracts"),
        requirements.get("experience_min_pct_of_budget"),
        requirements.get("experience_single_contract_min_pct"),
    )
    if not (
        codes or by_object or any(value is not None for value in (maximum, percentage, single_percentage))
    ):
        return []
    missing = []
    if not codes and not by_object:
        missing.append("UNSPSC codes or a similar-object criterion")
    if by_object and similar_contract_ids is None:
        missing.append("similar-contract selection")
    if maximum is None or maximum <= 0:
        missing.append("maximum contract count")
    if percentage is None:
        missing.append("experience percentage")
    if not requirements.get("official_budget"):
        if by_object:
            warnings.append(
                "Incomplete budget (zero or absent): minimum experience is COP 0; "
                "at least one similar contract is required."
            )
        else:
            missing.append("budget")
    if missing:
        return [Check("Experience", "Sufficient evidence", "; ".join(missing), None)]
    contracts = [
        contract
        for contract in company["experiencia"]
        if (
            contract["id"] in similar_contract_ids
            if by_object
            else codes & {str(code).removeprefix("V1.") for code in contract["codigos_unspsc"]}
        )
    ]
    checks = _experience_amount_checks(requirements, contracts, strict)
    if not strict:
        warnings.append("Experience compared in nominal pesos, without per-year SMMLV conversion.")
    return checks


def verify_requirements(
    requirements: dict | None,
    company: dict,
    similar_contract_ids: list[str] | None = None,
    strict: bool = False,
) -> dict:
    """Evaluate explicit rules; a proven failure takes priority over unknown evidence."""
    base = {
        key: (requirements or {}).get(key) for key in ("reference", "entity", "modality", "official_budget")
    }
    checks: list[Check] = []
    warnings = list((requirements or {}).get("observations") or [])

    def result(verdict: Verdict, summary: str, **extra: Any) -> dict:
        return {
            **base,
            "verdict": verdict.value,
            "summary": summary,
            "checks": [asdict(check) for check in checks],
            "warnings": warnings,
            **extra,
        }

    if requirements is None:
        checks.append(Check("Extraction", "Tender requirements", "Undetermined", None))
        warnings.append("Review the PDF, connection and model configuration.")
        return result(Verdict.MANUAL_REVIEW, "Could not determine the PDF requirements.")
    if requirements.get("is_direct_contracting"):
        return result(Verdict.NOT_APPLICABLE, "Direct hiring without a competitive call for bids.")
    if requirements.get("is_asset_sale"):
        return result(Verdict.NOT_APPLICABLE, "The entity sells assets; the company would be a buyer.")
    errors = validate_company(company)
    if errors:
        checks.append(Check("Company profile", "Valid data", "; ".join(errors), None))
        return result(Verdict.MANUAL_REVIEW, "Could not validate the company profile.")
    budget = requirements.get("official_budget")
    indicators = _raw_financial_indicators(company, budget)
    checks.extend(_financial_checks(requirements, indicators))
    checks.extend(_experience_checks(requirements, company, similar_contract_ids, strict, warnings))
    failures = [check.requirement for check in checks if check.passed is False]
    pending = [check.requirement for check in checks if check.passed is None]
    if failures:
        verdict, summary = Verdict.NOT_ELIGIBLE, "Does not meet: " + ", ".join(failures) + "."
    elif pending:
        verdict, summary = Verdict.MANUAL_REVIEW, "Could not determine: " + ", ".join(pending) + "."
    else:
        verdict, summary = Verdict.ELIGIBLE, "Meets all verified requirements."
    return result(verdict, summary, indicators=_compute_financial_indicators(company, budget))
