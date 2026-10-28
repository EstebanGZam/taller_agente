"""Analyze tender batches and report observed accuracy, cost and latency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import statistics
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

from secop_agent import settings
from secop_agent.eligibility import validate_company, verify_requirements
from secop_agent.eligibility_agent import EligibilityAgent
from secop_agent.tender_documents import normalize_model

logger = logging.getLogger(__name__)
CSV_COLUMNS = [
    "n",
    "file",
    "reference",
    "entity",
    "modality",
    "official_budget",
    "verdict",
    "expected_verdict",
    "correct",
    "summary",
    "tools_used",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_s",
    "model",
]


def load_company(path: str | Path = settings.MULTIPURPOSE_COMPANY_FILE) -> dict:
    """Load and validate the course-format company JSON before making model calls."""
    company = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_company(company)
    if errors:
        raise ValueError("Invalid company: " + " ".join(errors))
    return company


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deduplicate_pdfs(paths: Iterable[str | Path]) -> list[Path]:
    """Keep one path per SHA-256, preferring canonical names over copy suffixes."""
    unique = {}
    for path in sorted(map(Path, paths), key=lambda item: ("(" in item.stem, str(item))):
        unique.setdefault(file_digest(path), path)
    return sorted(unique.values(), key=lambda item: (item.name, str(item)))


def key_by_content(company: dict) -> dict[str, dict]:
    """Apply the key only to the reference company and PDFs with known content."""
    if company != load_company():
        return {}
    key = json.loads(settings.ELIGIBILITY_KEY_FILE.read_text(encoding="utf-8"))
    return {file_digest(settings.COURSE_DATA_DIR / process["file"]): process for process in key["processes"]}


def summarize(
    results: list[dict], rows: list[dict], model: str, cost_complete: bool = True, strict: bool = False
) -> dict:
    evaluated = [row for row in rows if row["correct"] is not None]
    correct = sum(row["correct"] for row in evaluated)
    total_cost = round(sum(result["metrics"]["cost_usd"] for result in results), 6)
    return {
        "team_name": settings.team_name(),
        "repository_url": settings.repository_url(),
        "processes": len(results),
        "counts": dict(Counter(result["verdict"] for result in results)),
        "correct": correct if evaluated else None,
        "evaluated": len(evaluated),
        "accuracy_pct": round(100 * correct / len(evaluated), 2) if evaluated else None,
        "total_cost_usd": total_cost,
        "mean_cost_usd": total_cost / len(results) if results else 0,
        "cost_complete": cost_complete,
        "p50_latency_s": statistics.median(result["metrics"]["latency_s"] for result in results)
        if results
        else 0,
        "input_tokens": sum(result["metrics"]["input_tokens"] for result in results),
        "output_tokens": sum(result["metrics"]["output_tokens"] for result in results),
        "model": normalize_model(model),
        "strict": strict,
        "key_comparison": "nominal pesos" if evaluated else None,
    }


def analyze_batch(
    paths: Iterable[str | Path],
    company: dict,
    model: str | None = None,
    workers: int = 3,
    strict: bool = False,
) -> dict:
    """Analyze unique PDFs and retain successful agent conversations for follow-up."""
    errors = validate_company(company)
    if errors:
        raise ValueError("Invalid company: " + " ".join(errors))
    if workers < 1:
        raise ValueError("Worker count must be greater than zero.")
    paths = deduplicate_pdfs(paths)
    if not paths:
        raise ValueError("No PDFs were found to analyze.")
    key = key_by_content(company)
    model = normalize_model(model)

    def analyze_one(path: Path) -> tuple[EligibilityAgent | None, dict]:
        agent = None
        started = time.perf_counter()
        try:
            agent = EligibilityAgent(path, company, model=model, strict=strict)
            return agent, agent.analyze()
        except Exception as exc:
            logger.warning("Document analysis failed (%s).", type(exc).__name__)
            fallback = verify_requirements(None, company)
            return agent, {
                **{
                    field: fallback[field]
                    for field in ("reference", "entity", "verdict", "summary", "checks", "warnings")
                },
                "file": path.name,
                "explanation": "",
                "tools_used": [],
                "metrics": {
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0,
                    "latency_s": round(time.perf_counter() - started, 3),
                },
                "model": model,
                "run_id": None,
                "trace_url": None,
            }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(pool.map(analyze_one, paths))
    results, agents, rows = [], {}, []
    complete = True
    for number, (path, (agent, result)) in enumerate(zip(paths, pairs, strict=True), 1):
        expected = key.get(file_digest(path))
        result = {
            **result,
            "n": number,
            "expected_verdict": expected["expected_verdict"] if expected else None,
            "correct": result["verdict"] == expected["expected_verdict"] if expected else None,
        }
        results.append(result)
        successful = agent is not None and agent.last_analysis is not None
        if successful:
            agents[str(number)] = agent
        complete = complete and successful and agent.cost_complete
        row = {column: result.get(column) for column in CSV_COLUMNS}
        row.update(result["metrics"])
        requirements = agent.requirements if agent is not None else None
        row.update({field: (requirements or {}).get(field) for field in ("modality", "official_budget")})
        row["tools_used"] = ", ".join(result["tools_used"])
        rows.append(row)
    return {
        "results": results,
        "agents": agents,
        "rows": rows,
        "summary": summarize(results, rows, model, complete, strict),
    }


def summary_text(summary: dict) -> str:
    """Format the team attribution and observed batch metrics as plain text."""
    accuracy = (
        f"{summary['correct']}/{summary['evaluated']} · accuracy {summary['accuracy_pct']:.2f}%"
        if summary["evaluated"]
        else "Accuracy: no applicable key"
    )
    counts = ", ".join(f"{verdict}: {count}" for verdict, count in summary["counts"].items())
    return (
        f"TEAM_NAME: {summary['team_name']}\nRepository: {summary['repository_url']}\n"
        f"Processes: {summary['processes']} · {accuracy}\nVerdicts: {counts}\n"
        f"Total cost USD: {summary['total_cost_usd']:.6f}\n"
        f"Mean cost per process USD: {summary['mean_cost_usd']:.6f}\n"
        f"Cost complete: {summary['cost_complete']}\n"
        f"p50 latency: {summary['p50_latency_s']:.3f} s\n"
        f"Tokens: input {summary['input_tokens']}, output {summary['output_tokens']}\n"
        f"Model: {summary['model']}\nStrict SMMLV: {summary['strict']}\n"
        "Reference key uses nominal pesos."
    )


def write_results(batch: dict, output_dir: str | Path) -> Path:
    """Write the specified CSV columns and the full report as Markdown and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eligibility_processes.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(batch["rows"])
    (output_dir / "eligibility_processes.json").write_text(
        json.dumps(
            {key: batch[key] for key in ("summary", "results", "rows")},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    def cell(value: object) -> str:
        return escape(str(value if value is not None else "")).replace("|", "\\|").replace("\n", " ")

    table = ["| " + " | ".join(CSV_COLUMNS) + " |", "| " + " | ".join(["---"] * len(CSV_COLUMNS)) + " |"]
    table.extend(
        "| " + " | ".join(cell(row[column]) for column in CSV_COLUMNS) + " |" for row in batch["rows"]
    )
    (output_dir / "eligibility_processes.md").write_text(
        "# Tender eligibility\n\n"
        + escape(summary_text(batch["summary"]))
        + "\n\n"
        + "\n".join(table)
        + "\n",
        encoding="utf-8",
    )
    return csv_path


def main(argv: list[str] | None = None) -> int:
    settings.load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", type=Path, default=settings.MULTIPURPOSE_COMPANY_FILE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pdf-dir", type=Path, default=settings.COURSE_DATA_DIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=settings.RESULTS_DIR)
    args = parser.parse_args(argv)
    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is missing; configure it in the environment or .env.")
    try:
        batch = analyze_batch(
            args.pdf_dir.glob("*.pdf"), load_company(args.company), args.model, args.workers, args.strict
        )
        write_results(batch, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(summary_text(batch["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
