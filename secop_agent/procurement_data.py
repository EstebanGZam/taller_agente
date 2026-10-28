"""SECOP II procurement data and company financial indicators: the data layer of the agents.

Pure Python functions that can be tested without a language model. This is the English port of the
course starter `herramientas.py`; its numeric behaviour is identical because the reference answers
in `datos/conjunto_referencia.json` depend on it. Raw SECOP field names (Spanish, as returned by the
open-data API) and the company JSON keys are data contracts and stay unchanged.

Run ``python -m secop_agent.procurement_data`` for an offline smoke test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from secop_agent.settings import REFERENCE_COMPANY_FILE, SAMPLE_PROCESSES_FILE

logger = logging.getLogger(__name__)

API_URL = "https://www.datos.gov.co/resource/p6dx-8zbt.json"  # SECOP II - Procesos de Contratación
API_FIELDS = (
    "entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "nombre_de_la_unidad_de",
    "ciudad_de_la_unidad_de",
    "id_del_proceso",
    "referencia_del_proceso",
    "descripci_n_del_procedimiento",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "precio_base",
    "duracion",
    "unidad_de_duracion",
    "fecha_de_publicacion_del",
    "estado_resumen",
    "nombre_del_proveedor",
    "valor_total_adjudicacion",
    "respuestas_al_procedimiento",
    "urlproceso",
)
API_TIMEOUT_S = 30

DATA_MODE_ENV = "DATA_MODE"
SAMPLE_MODE = "sample"  # The 20 local processes in datos/secop_muestra.json (reference set).
API_MODE = "api"  # Live SECOP II data from datos.gov.co.

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class ProcessFilters:
    """Optional search filters. Text filters are case- and accent-insensitive substrings."""

    entity: str | None = None
    department: str | None = None
    city: str | None = None
    modality: str | None = None
    text: str | None = None
    min_budget: float | None = None
    max_budget: float | None = None
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None  # YYYY-MM-DD


def normalize_text(value: Any) -> str:
    """Lower case without accents, for comparing Spanish text."""
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def to_number(value: Any) -> float:
    """SECOP numbers arrive as strings; anything unparsable counts as 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def publication_date(process: dict) -> str:
    """Publication date as YYYY-MM-DD ('' when missing)."""
    return (process.get("fecha_de_publicacion_del") or "")[:10]


def summarize(processes: list[dict]) -> dict:
    """Count and budget statistics of a list of raw SECOP processes."""
    budgets = [to_number(p.get("precio_base")) for p in processes]
    positive = [b for b in budgets if b > 0]
    return {
        "count": len(processes),
        "total_budget": sum(budgets),
        "average_budget": round(sum(budgets) / len(budgets), 2) if budgets else 0,
        "zero_budget_processes": len(budgets) - len(positive),
    }


def load_sample_processes(path: Path = SAMPLE_PROCESSES_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains(needle: str | None, haystack: Any) -> bool:
    return not needle or normalize_text(needle) in normalize_text(haystack)


def matches(process: dict, filters: ProcessFilters) -> bool:
    """True when a raw SECOP process satisfies every filter that is set."""
    cities = f"{process.get('ciudad_entidad') or ''} {process.get('ciudad_de_la_unidad_de') or ''}"
    if not (
        _contains(filters.entity, process.get("entidad"))
        and _contains(filters.department, process.get("departamento_entidad"))
        and _contains(filters.city, cities)
        and _contains(filters.modality, process.get("modalidad_de_contratacion"))
        and _contains(filters.text, process.get("descripci_n_del_procedimiento"))
    ):
        return False
    budget = to_number(process.get("precio_base"))
    if filters.min_budget is not None and budget < filters.min_budget:
        return False
    if filters.max_budget is not None and budget > filters.max_budget:
        return False
    date = publication_date(process)
    if filters.date_from and date < filters.date_from:
        return False
    return not (filters.date_to and date > filters.date_to)


def filter_sample(filters: ProcessFilters, processes: list[dict] | None = None) -> list[dict]:
    """Processes of the local sample (or of `processes`) that match `filters`, in file order."""
    candidates = load_sample_processes() if processes is None else processes
    return [p for p in candidates if matches(p, filters)]


def soql_literal(value: Any) -> str:
    """Upper-cased SoQL string content with single quotes escaped (prevents query injection)."""
    return str(value).replace("'", "''").upper()


def build_soql_where(filters: ProcessFilters) -> str | None:
    """The `$where` clause equivalent to `matches`, or None without filters."""
    conditions = []
    if filters.entity:
        conditions.append(f"upper(entidad) like '%{soql_literal(filters.entity)}%'")
    if filters.department:
        conditions.append(f"upper(departamento_entidad) like '%{soql_literal(filters.department)}%'")
    if filters.city:
        city = soql_literal(filters.city)
        conditions.append(
            f"(upper(ciudad_entidad) like '%{city}%' OR upper(ciudad_de_la_unidad_de) like '%{city}%')"
        )
    if filters.modality:
        conditions.append(f"upper(modalidad_de_contratacion) like '%{soql_literal(filters.modality)}%'")
    if filters.text:
        conditions.append(f"upper(descripci_n_del_procedimiento) like '%{soql_literal(filters.text)}%'")
    if filters.min_budget is not None:
        conditions.append(f"precio_base >= {float(filters.min_budget)}")
    if filters.max_budget is not None:
        conditions.append(f"precio_base <= {float(filters.max_budget)}")
    if filters.date_from:
        conditions.append(f"fecha_de_publicacion_del >= '{soql_literal(filters.date_from)}T00:00:00'")
    if filters.date_to:
        conditions.append(f"fecha_de_publicacion_del <= '{soql_literal(filters.date_to)}T23:59:59'")
    return " AND ".join(conditions) or None


def query_api(filters: ProcessFilters, limit: int) -> list[dict]:
    """Most recent SECOP II processes from datos.gov.co that match `filters`."""
    params: dict[str, Any] = {
        "$select": ",".join(API_FIELDS),
        "$limit": limit,
        "$order": "fecha_de_publicacion_del DESC",
    }
    where = build_soql_where(filters)
    if where:
        params["$where"] = where
    headers = {}
    if os.getenv("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    response = requests.get(API_URL, params=params, headers=headers, timeout=API_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def data_mode() -> str:
    """`api` for live data; anything else means the local sample."""
    return API_MODE if os.getenv(DATA_MODE_ENV, SAMPLE_MODE).strip().lower() == API_MODE else SAMPLE_MODE


def clamp_limit(limit: int | None, default: int = DEFAULT_LIMIT) -> int:
    return max(1, min(int(limit or default), MAX_LIMIT))


def search_processes(
    entity: str | None = None,
    department: str | None = None,
    city: str | None = None,
    modality: str | None = None,
    text: str | None = None,
    min_budget: float | None = None,
    max_budget: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """Search SECOP II procurement processes.

    Uses the local sample (`DATA_MODE=sample`, the default) or the live API (`DATA_MODE=api`).
    Returns ``{"summary": ..., "processes": [...]}`` with the raw SECOP records, or
    ``{"error": ...}`` when the API cannot be reached.
    """
    filters = ProcessFilters(
        entity, department, city, modality, text, min_budget, max_budget, date_from, date_to
    )
    limit = clamp_limit(limit)
    if data_mode() == API_MODE:
        try:
            processes = query_api(filters, limit)
        except requests.RequestException as exc:
            logger.warning("SECOP API query failed: %s", exc)
            return {"error": f"Could not query the SECOP API: {exc}"}
    else:
        processes = filter_sample(filters)[:limit]
    return {"summary": summarize(processes), "processes": processes}


def load_company(path: Path) -> dict:
    """Read a company profile JSON (course format, Spanish keys)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_company_profile() -> dict:
    """The reference company of the course: RUP, UNSPSC codes, financial statements and experience."""
    return load_company(REFERENCE_COMPANY_FILE)


def compute_financial_indicators(company: dict, process_budget: float | None = None) -> dict:
    """RUP financial and organisational indicators of `company`.

    With a (non-zero) process budget it also returns the share of that budget covered by the
    company's working capital. Rounding matches the course's `herramientas.calcular_indicadores`.
    """
    statements = company["estados_financieros"]

    def ratio(numerator: str, denominator: str, digits: int, scale: float = 1.0) -> float | None:
        # None instead of ZeroDivisionError for uploaded profiles with a zero divisor.
        divisor = statements[denominator]
        return round(scale * statements[numerator] / divisor, digits) if divisor else None

    indicators: dict[str, Any] = {
        "liquidity_ratio": ratio("activo_corriente", "pasivo_corriente", 2),
        "debt_ratio_pct": ratio("pasivo_total", "activo_total", 1, scale=100),
        # No interest expense means coverage is undetermined, which RUP rules treat as passing.
        "interest_coverage_ratio": (
            ratio("utilidad_operacional", "gastos_de_intereses", 2)
            if statements["gastos_de_intereses"]
            else UNDETERMINED
        ),
        "working_capital": statements["activo_corriente"] - statements["pasivo_corriente"],
        "return_on_equity": ratio("utilidad_operacional", "patrimonio", 3),
        "return_on_assets": ratio("utilidad_operacional", "activo_total", 3),
        "financial_statements_date": statements["corte"],
    }
    if process_budget:
        indicators["working_capital_pct_of_budget"] = round(
            100 * indicators["working_capital"] / float(process_budget), 1
        )
    return indicators


def main(argv: list[str] | None = None) -> int:
    """Offline smoke test: indicators of the reference company and a sample search."""
    parser = argparse.ArgumentParser(description="Offline smoke test of the SECOP data tools.")
    parser.add_argument("--modality", default="licitaci", help="Modality substring to search for.")
    args = parser.parse_args(argv)
    print("Financial indicators:", compute_financial_indicators(get_company_profile()))
    print(
        f"Search modality={args.modality!r} ({data_mode()} mode):",
        search_processes(modality=args.modality)["summary"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
