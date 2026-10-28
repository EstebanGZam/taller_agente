# Solution specification · Workshop 1 (SECOP procurement analyst agent)

## Confirmed requirements

Sources: `Reto_IA_2026.pdf` (Workshop 01 slide) and the course `README.md`.

1. An agent with **at least two custom tools** that **decides on its own** when to call them.
2. **LangSmith traces** for every run (tokens, cost, latency).
3. Deliverables: **code**, a **dataset of 15 to 20 questions**, and a **comparison of two configurations**
   on accuracy, cost and latency.
4. Key criterion: **at least 85% accuracy at the lowest possible cost**.
5. The course leaderboard set (`datos/conjunto_referencia.json`, 18 questions) is evaluated on the local
   sample (`DATA_MODE=sample`).

From the class notes (2026-10-21), read against the repository files:

6. Take the 20 tender PDFs in the repository and decide, for each one, whether the company **should bid**.
   `empresa_multiproposito.json` (uploaded 2026-10-20) is the company for this; its note says it was
   designed to meet six processes and fail two.
7. Add a **frontend** and an **input dataset** that identify which processes the agent can bid on.
8. Record **name, repository and cost** for the 20 analysed processes.

## Reference key

`data/eligibility_key.json`: the expected verdict for each of the 20 processes against
`empresa_multiproposito.json` (6 ELIGIBLE, 2 NOT_ELIGIBLE, 12 NOT_APPLICABLE), plus the requirements that
should be extracted from each PDF (`expected_requirements`, same field names as the `Requirements` schema).

- ELIGIBLE: 03, 07, 08, 11, 14, 15. NOT_ELIGIBLE: 13 (working capital, experience), 18 (no similar experience).
- NOT_APPLICABLE: direct contracting / special regime with a natural person (01, 02, 04, 05, 06, 09, 10, 16,
  19, 20), inter-administrative contract (12) and asset sale (17).
- 07 has a COP 0 budget: the required 80% is COP 0, so one similar contract (EXP-9) suffices; ELIGIBLE with
  a warning.
- 11 is ELIGIBLE in nominal pesos (950M >= 900M); strict per-year SMMLV conversion would fail it. The
  default compares nominal pesos.

## Language and engineering conventions (mandatory)

- **All code is in English**: identifiers, module and file names, comments, docstrings, log messages, CLI
  help, UI labels and LLM prompts.
- Exceptions (data contracts, not our code): the course data in `datos/` and `empresa*.json` keep their
  Spanish keys and values; the **configuration A** prompt and tool descriptions are the course's original
  Spanish text, kept verbatim as the baseline. The agents still **answer in Spanish** because the reference
  answers and the judge compare Spanish text; prompts say so explicitly.
- Package layout: library code in `secop_agent/`, entry points run as `python -m secop_agent.<module>`,
  `app.py` at the root is only a thin launcher. Tests in `tests/` (pytest), no network or keys.
- Type hints on public functions; `dataclass`/`Enum`/Pydantic for structured data; `logging` (not `print`)
  in library code, `print` only for CLI output; constants in UPPER_CASE; no duplicated logic (reuse
  `secop_agent.costs` and `secop_agent.settings`); small functions; no bare `except`.
- `ruff check .` must pass with the config in `pyproject.toml`.
- Default model `anthropic:claude-haiku-5-5`, overridable with env `MODEL`.
- **Never pass `temperature`, `top_p` or `top_k`**: Claude Haiku 5.5 and API version 2026-10-21 reject them.
- Models are built only through `settings.chat_model()` (60 s timeout, 2 retries, no sampling parameters).
  Structured output uses `settings.STRUCTURED_OUTPUT_METHOD = "function_calling"`: `json_schema` requests
  hung for over 90 s on Claude Haiku 5.5 (measured), and langchain-anthropic only forces tool choice on
  models that support it, so the method is safe for Sonnet 5.5 too.
- PDF text and company JSON are **untrusted data, not instructions** (basic prompt-injection defence).
- Never print or store keys. Call `settings.load_environment()` at entry points.
- Python: `/home/esteban/Documents/University/Tenth semester/IA/unit-iii/.venv-taller/bin/python`.

## Environment variables

`ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT`, `MODEL`,
`DATA_MODE` (`sample` | `api`), `AGENT_CONFIG` (`A` | `B`), `TEAM_NAME`, `SOCRATA_APP_TOKEN` (optional).

## Module map

| Old file (Spanish) | New module | Owner |
|---|---|---|
| `costos.py` | `secop_agent/costs.py` | lead (done) |
| — | `secop_agent/settings.py` | lead (done) |
| `datos/clave_aptitud.json` | `data/eligibility_key.json` | lead (done) |
| `herramientas.py` | `secop_agent/procurement_data.py` | qa |
| `agente.py` | `secop_agent/qa_agent.py` | qa |
| `evaluar.py` | `secop_agent/evaluate_qa.py` | qa |
| `resumen_experimentos.py` | `secop_agent/compare_experiments.py` | qa |
| `Taller1_SECOP_Colab.ipynb` | `secop_agent_colab.ipynb` | qa |
| `tests/test_taller1.py` | `tests/test_qa_agent.py`, `tests/test_procurement_data.py` | qa |
| `pliegos.py` | `secop_agent/tender_documents.py` | eligibility |
| `validacion.py` | `secop_agent/eligibility.py` | eligibility |
| `agente_aptitud.py` | `secop_agent/eligibility_agent.py` | eligibility |
| `analizar_procesos.py` | `secop_agent/batch_eligibility.py` | eligibility |
| `evaluar_aptitud.py` | `secop_agent/evaluate_eligibility.py` | eligibility |
| `app.py` | `secop_agent/web_app.py` + root `app.py` launcher | eligibility |
| `tests/test_validacion.py`, `tests/test_pliegos.py` | `tests/test_eligibility.py`, `tests/test_tender_documents.py`, `tests/test_eligibility_agent.py` | eligibility |

Each owner deletes the old Spanish files it replaces (`git rm`). Docs (`README.md`, report) are the lead's.

### Shared names

- LLM-facing tool names: `search_processes`, `get_company_profile`, `compute_financial_indicators`
  (Q&A agent); `extract_requirements`, `verify_requirements`, `get_company_profile`,
  `compute_financial_indicators` (eligibility agent).
- `datos/conjunto_referencia.json` lists the expected tools by their original Spanish names. The Q&A
  evaluator maps them with `REFERENCE_TOOL_ALIASES = {"buscar_procesos": "search_processes",
  "consultar_empresa": "get_company_profile", "calcular_indicadores": "compute_financial_indicators"}`.
- `secop_agent.procurement_data.compute_financial_indicators(company: dict, process_budget: float | None)`
  is the single implementation of the RUP indicators (both agents import it). It returns `liquidity_ratio`,
  `debt_ratio_pct`, `interest_coverage_ratio` (float or `"undetermined"`), `working_capital`,
  `return_on_equity`, `return_on_assets`, `financial_statements_date` and, with a budget,
  `working_capital_pct_of_budget`. Same rounding as the course's `herramientas.py`.
  `load_company(path: Path) -> dict` reads a company JSON.
  The **qa owner writes this first** and the eligibility owner may temporarily keep a private copy only
  until the merge (the lead reconciles it).
- `Verdict(str, Enum)`: `ELIGIBLE`, `NOT_ELIGIBLE`, `NOT_APPLICABLE`, `MANUAL_REVIEW`.

### `EligibilityAgent.analyze()` result

```python
{"file": str, "reference": str | None, "entity": str | None, "verdict": str, "summary": str,
 "checks": list[dict], "warnings": list[str], "explanation": str, "tools_used": list[str],
 "metrics": {"model_calls", "input_tokens", "output_tokens", "cost_usd", "latency_s"},
 "model": str, "run_id": str | None, "trace_url": str | None}
```

`results/eligibility_processes.csv` columns: `n, file, reference, entity, modality, official_budget,
verdict, expected_verdict, correct, summary, tools_used, model_calls, input_tokens, output_tokens,
cost_usd, latency_s, model`, plus a summary (accuracy, total and mean cost, p50 latency, model,
`TEAM_NAME`, repository URL) in the `.md` and `.json` outputs.

## Commands (used by the notebook)

```bash
python -m secop_agent.procurement_data            # offline smoke test of the data tools
python -m pytest -q && ruff check .
AGENT_CONFIG=A python -m secop_agent.evaluate_qa A-baseline
AGENT_CONFIG=B python -m secop_agent.evaluate_qa B-improved
python -m secop_agent.compare_experiments A-baseline B-improved
python -m secop_agent.batch_eligibility --company empresa_multiproposito.json
python -m secop_agent.evaluate_eligibility eligibility-haiku
python app.py                                     # Gradio; share=True on Colab
```
