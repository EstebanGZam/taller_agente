"""Side-by-side comparison of Q&A agent configurations from `secop_agent.evaluate_qa` experiments.

Usage::

    python -m secop_agent.compare_experiments A-baseline B-improved [...]

Reads ``results/qa_experiment_<prefix>.json`` and writes ``results/qa_comparison.md``. With a
LANGSMITH_API_KEY it fills in LangSmith's cost when the local file lacks it, and when there is no
local file for a prefix it builds a partial summary from the most recent experiment with that prefix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from secop_agent.evaluate_qa import CORRECT_ANSWER_KEY, COST_KEY, TOOL_USAGE_KEY, experiment_path
from secop_agent.settings import RESULTS_DIR, load_environment

logger = logging.getLogger(__name__)

TARGET_ACCURACY_PCT = 85.0  # The workshop's accuracy criterion.
COMPARISON_FILE = "qa_comparison.md"


def _feedback_mean(stats: dict | None, key: str) -> float | None:
    """Mean of one feedback key in LangSmith's `feedback_stats`."""
    value = (stats or {}).get(key)
    if isinstance(value, dict):
        value = value.get("avg", value.get("mean"))
    return float(value) if value is not None else None


def _as_pct(value: float | None) -> float | None:
    return round(100 * value, 1) if value is not None else None


def _read_langsmith_project(prefix: str, experiment: str | None) -> Any:
    """The LangSmith project of `experiment`, or of the most recent experiment named `prefix…`."""
    from langsmith import Client

    client = Client()
    if experiment:
        return client.read_project(project_name=experiment, include_stats=True)
    candidates = [
        p
        for p in client.list_projects(name_contains=prefix, include_stats=True)
        if (p.name or "").startswith(prefix)
    ]
    return max(candidates, key=lambda p: p.start_time) if candidates else None


def summary_from_project(prefix: str, project: Any) -> dict:
    """A partial experiment summary built from LangSmith's project statistics."""
    metadata = (project.extra or {}).get("metadata", {})
    feedback = project.feedback_stats or {}
    runs = project.run_count or 0
    cost_per_question = _feedback_mean(feedback, COST_KEY)
    p50 = project.latency_p50.total_seconds() if project.latency_p50 else None
    return {
        "prefix": prefix,
        "experiment": project.name,
        "agent_config": metadata.get("agent_config", "?"),
        "model": metadata.get("model", "?"),
        "change": project.description or "",
        "questions": runs,
        "tool_usage_accuracy": _as_pct(_feedback_mean(feedback, TOOL_USAGE_KEY)),
        "answer_accuracy": _as_pct(_feedback_mean(feedback, CORRECT_ANSWER_KEY)),
        "combined_accuracy": None,
        "input_tokens": project.prompt_tokens,
        "output_tokens": project.completion_tokens,
        "total_cost_usd": round(cost_per_question * runs, 6)
        if cost_per_question is not None and runs
        else None,
        "cost_per_question_usd": cost_per_question,
        "cost_per_correct_answer_usd": None,
        "langsmith_cost_usd": float(project.total_cost) if project.total_cost is not None else None,
        "latency_p50_s": round(p50, 2) if p50 is not None else None,
        "latency_p95_s": None,
    }


def enrich_from_langsmith(prefix: str, local: dict | None) -> dict | None:
    """Complete `local` with LangSmith's cost, or build a summary when there is no local file."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return local
    try:
        project = _read_langsmith_project(prefix, (local or {}).get("experiment"))
    except Exception as exc:  # noqa: BLE001  (LangSmith is optional here)
        logger.warning("Could not query LangSmith for %s: %s", prefix, exc)
        return local
    if project is None:
        return local
    if local is None:
        return summary_from_project(prefix, project)
    if local.get("langsmith_cost_usd") is None and project.total_cost is not None:
        local["langsmith_cost_usd"] = float(project.total_cost)
    return local


def load_experiment(prefix: str, results_dir: Path = RESULTS_DIR) -> dict | None:
    path = experiment_path(prefix, results_dir)
    local = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return enrich_from_langsmith(prefix, local)


def _fmt(value: Any, template: str = "{}", suffix: str = "") -> str:
    return "—" if value is None else template.format(value) + suffix


def verdict(experiments: list[dict]) -> str:
    """One line: which configurations meet the 85% criterion and which is cheapest per correct answer."""
    passing = [e for e in experiments if (e.get("answer_accuracy") or 0) >= TARGET_ACCURACY_PCT]
    names = ", ".join(f"{e['prefix']} ({e['answer_accuracy']}%)" for e in passing) or "none"
    text = f"Meeting the {TARGET_ACCURACY_PCT:.0f}% correct-answer criterion: {names}."
    priced = [e for e in experiments if e.get("cost_per_correct_answer_usd") is not None]
    if priced:
        cheapest = min(priced, key=lambda e: e["cost_per_correct_answer_usd"])
        text += (
            f" Cheapest per correct answer: {cheapest['prefix']} "
            f"(USD {cheapest['cost_per_correct_answer_usd']:.6f})."
        )
    if passing:
        chosen = min(passing, key=lambda e: e.get("cost_per_question_usd") or float("inf"))
        text += (
            f" Recommended: {chosen['prefix']}"
            f" (>= {TARGET_ACCURACY_PCT:.0f}% at the lowest cost per question)."
        )
    return text


ROWS: list[tuple[str, Callable[[dict], str]]] = [
    ("LangSmith experiment", lambda e: f"`{e.get('experiment') or e['prefix']}`"),
    ("Configuration", lambda e: e.get("agent_config", "?")),
    ("Model", lambda e: f"`{e.get('model', '?')}`"),
    ("Change", lambda e: e.get("change") or "—"),
    ("Questions", lambda e: _fmt(e.get("questions"))),
    ("Tool usage", lambda e: _fmt(e.get("tool_usage_accuracy"), suffix=" %")),
    ("Correct answer", lambda e: _fmt(e.get("answer_accuracy"), suffix=" %")),
    ("Both correct", lambda e: _fmt(e.get("combined_accuracy"), suffix=" %")),
    (
        "Input / output tokens",
        lambda e: f"{_fmt(e.get('input_tokens'), '{:,}')} / {_fmt(e.get('output_tokens'), '{:,}')}",
    ),
    ("Total cost (ours)", lambda e: _fmt(e.get("total_cost_usd"), "USD {:.6f}")),
    ("Cost per question", lambda e: _fmt(e.get("cost_per_question_usd"), "USD {:.6f}")),
    ("Cost per correct answer", lambda e: _fmt(e.get("cost_per_correct_answer_usd"), "USD {:.6f}")),
    ("Total cost (LangSmith)", lambda e: _fmt(e.get("langsmith_cost_usd"), "USD {:.6f}")),
    ("Latency p50", lambda e: _fmt(e.get("latency_p50_s"), "{:.2f}", " s")),
    ("Latency p95", lambda e: _fmt(e.get("latency_p95_s"), "{:.2f}", " s")),
]


def comparison_table(experiments: list[dict]) -> str:
    """Markdown table with one column per configuration."""
    lines = [
        "| | " + " | ".join(e["prefix"] for e in experiments) + " |",
        "|---|" + "---|" * len(experiments),
    ]
    for label, render in ROWS:
        lines.append(
            f"| {label} | " + " | ".join(str(render(e)).replace("|", "/") for e in experiments) + " |"
        )
    return "\n".join(lines)


def build_report(experiments: list[dict]) -> str:
    return (
        "# Q&A agent configurations · Workshop 1\n\n"
        "Reference set (18 questions, `DATA_MODE=sample`). Our cost comes from `secop_agent.costs` "
        "(Anthropic's public rates); LangSmith may not know the model's rate.\n\n"
        + comparison_table(experiments)
        + "\n\n**Verdict:** "
        + verdict(experiments)
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Q&A agent experiments side by side.")
    parser.add_argument("prefixes", nargs="+", help="Experiment prefixes, e.g. A-baseline B-improved.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    load_environment()
    logging.basicConfig(level=logging.WARNING)
    experiments = []
    for prefix in args.prefixes:
        experiment = load_experiment(prefix, args.results_dir)
        if experiment is None:
            print(
                f"No results for {prefix!r} (run: python -m secop_agent.evaluate_qa {prefix}).",
                file=sys.stderr,
            )
            continue
        experiments.append(experiment)
    if not experiments:
        print("No experiments to compare.", file=sys.stderr)
        return 1
    report = build_report(experiments)
    path = args.results_dir / COMPARISON_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
