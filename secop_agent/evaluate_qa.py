"""Evaluate the Q&A agent in LangSmith on the course's reference question set.

Usage::

    AGENT_CONFIG=A python -m secop_agent.evaluate_qa A-baseline
    AGENT_CONFIG=B python -m secop_agent.evaluate_qa B-improved

Each example records the tools used, tokens, cost (`secop_agent.costs`) and latency in its outputs.
Two evaluators score it: tool usage (exact, against the expected tools) and answer correctness (an
LLM judge). The aggregate is printed and saved to ``results/qa_experiment_<prefix>.json``, which
`secop_agent.compare_experiments` reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from secop_agent import qa_agent
from secop_agent.costs import CostTracker
from secop_agent.procurement_data import DATA_MODE_ENV, SAMPLE_MODE, data_mode
from secop_agent.settings import (
    DEFAULT_MODEL,
    REFERENCE_QUESTIONS_FILE,
    RESULTS_DIR,
    load_environment,
    model_from_env,
    team_name,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "secop-taller1-referencia"  # The course's dataset name, shared across teams.
JUDGE_MODEL_ENV = "JUDGE_MODEL"
REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY")
DEFAULT_MAX_CONCURRENCY = 2

# The dataset's example schema is the course's (Spanish keys); changing it would break the
# LangSmith dataset that already exists under DATASET_NAME.
QUESTION_KEY = "pregunta"
EXPECTED_ANSWER_KEY = "respuesta"
EXPECTED_TOOLS_KEY = "herramientas"

# The reference set lists the expected tools by their original Spanish names.
REFERENCE_TOOL_ALIASES = {
    "buscar_procesos": "search_processes",
    "consultar_empresa": "get_company_profile",
    "calcular_indicadores": "compute_financial_indicators",
}

TOOL_USAGE_KEY = "tool_usage"
CORRECT_ANSWER_KEY = "correct_answer"
COST_KEY = "cost_usd"
METRIC_KEYS = ("model_calls", "input_tokens", "output_tokens", "cost_usd", "cost_complete")

JUDGE_INSTRUCTIONS = (
    "You are a strict grader. Compare the ANSWER with the EXPECTED ANSWER to the QUESTION (both in "
    "Spanish). Figures must match; differences in formatting and rounding are acceptable. Do not "
    "penalise additional information when it is correct and does not contradict the expected answer. "
    "Give a one-sentence reason, then set correct to true or false."
)


class JudgeVerdict(BaseModel):
    """Structured output of the judge."""

    reason: str = Field(description="One sentence explaining the decision.")
    correct: bool = Field(description="True when the answer matches the expected answer.")


def judge_model() -> str:
    return os.getenv(JUDGE_MODEL_ENV, DEFAULT_MODEL)


def load_reference_cases(path: Path = REFERENCE_QUESTIONS_FILE) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_tool_names(names: list[str] | None) -> set[str]:
    """Map the reference set's Spanish tool names to the agent's English ones."""
    return {REFERENCE_TOOL_ALIASES.get(name, name) for name in names or []}


def run_agent(inputs: dict) -> dict:
    """LangSmith target: run the agent on one question, measuring tokens, cost and latency."""
    model = model_from_env()
    tracker = CostTracker(model)
    start = time.perf_counter()
    result = qa_agent.ask(inputs[QUESTION_KEY], model=model, callbacks=[tracker])
    latency = time.perf_counter() - start
    usage = tracker.summary()
    metrics = {key: usage[key] for key in METRIC_KEYS}
    return {**asdict(result), "metrics": {**metrics, "latency_s": round(latency, 3)}}


def tool_usage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """1 when every expected tool was used, or no tool at all when none was expected."""
    expected = canonical_tool_names(reference_outputs.get(EXPECTED_TOOLS_KEY))
    used = set(outputs.get("tools_used") or [])
    ok = expected.issubset(used) if expected else not used
    return {"key": TOOL_USAGE_KEY, "score": int(ok)}


_VERDICT_WORD = re.compile(r"\b(INCORRECTA?|CORRECTA?)\b")


def parse_verdict(verdict: Any) -> int | None:
    """1 for a correct verdict, 0 for an incorrect one, None when it cannot be read.

    Accepts a `JudgeVerdict`, the dict returned by ``with_structured_output(..., include_raw=True)``
    (falling back to the raw message text when parsing failed), or free text. In text the whole word
    is matched, because "INCORRECT" contains "CORRECT" as a substring.
    """
    if isinstance(verdict, JudgeVerdict):
        return int(verdict.correct)
    if isinstance(verdict, dict):
        if isinstance(verdict.get("correct"), bool):
            return int(verdict["correct"])
        if verdict.get("parsed") is not None:
            return parse_verdict(verdict["parsed"])
        raw = verdict.get("raw")
        verdict = getattr(raw, "content", raw)
    match = _VERDICT_WORD.search(qa_agent.message_text(verdict).upper())
    return None if match is None else int(match.group(1).startswith("CORRECT"))


_judge: Any = None


def get_judge() -> Any:
    """The judge model with structured output, built once."""
    global _judge
    if _judge is None:
        from langchain.chat_models import init_chat_model

        # No temperature: Claude Haiku 5.5 rejects it. json_schema avoids forced tool choice.
        # include_raw keeps the raw reply so an unparsable verdict can still be read as text.
        _judge = init_chat_model(judge_model()).with_structured_output(
            JudgeVerdict, method="json_schema", include_raw=True
        )
    return _judge


def judge_prompt(question: str, expected: str, answer: str) -> str:
    return f"{JUDGE_INSTRUCTIONS}\n\nQUESTION: {question}\nEXPECTED ANSWER: {expected}\nANSWER: {answer}"


def correct_answer(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: does the answer match the expected answer?"""
    verdict = get_judge().invoke(
        judge_prompt(
            inputs[QUESTION_KEY], reference_outputs[EXPECTED_ANSWER_KEY], outputs.get("answer") or ""
        )
    )
    score = parse_verdict(verdict)
    parsed = verdict.get("parsed") if isinstance(verdict, dict) else verdict
    if score is None:
        comment = f"Unreadable verdict: {str(verdict)[:200]}"
    else:
        comment = getattr(parsed, "reason", None)
    return {"key": CORRECT_ANSWER_KEY, "score": score or 0, "comment": comment}


def cost_usd(outputs: dict) -> dict:
    """Per-example cost as a feedback column (visible in LangSmith's comparison view)."""
    return {"key": COST_KEY, "score": (outputs.get("metrics") or {}).get("cost_usd")}


def percentile(values: list[float], pct: float) -> float | None:
    """Percentile with linear interpolation (`pct` between 0 and 100)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    if lower + 1 >= len(ordered):
        return ordered[lower]
    return ordered[lower] + (ordered[lower + 1] - ordered[lower]) * (position - lower)


def _round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def summarize_experiment(
    rows: list[dict],
    prefix: str,
    config: str,
    model: str,
    experiment: str | None = None,
    langsmith_cost: float | None = None,
) -> dict:
    """Aggregate the per-question rows into the experiment's metrics."""
    n = len(rows)

    def accuracy(key: str) -> float:
        return round(100 * sum(row[key] or 0 for row in rows) / n, 1) if n else 0.0

    latencies = [row["latency_s"] for row in rows if row.get("latency_s") is not None]
    cost = sum(row["cost_usd"] or 0 for row in rows)
    correct = sum(row[CORRECT_ANSWER_KEY] or 0 for row in rows)
    both = sum(1 for row in rows if row[TOOL_USAGE_KEY] == 1 and row[CORRECT_ANSWER_KEY] == 1)
    return {
        "prefix": prefix,
        "experiment": experiment,
        "agent_config": config,
        "model": model,
        "change": qa_agent.CONFIG_DESCRIPTIONS.get(config, ""),
        "questions": n,
        "tool_usage_accuracy": accuracy(TOOL_USAGE_KEY),
        "answer_accuracy": accuracy(CORRECT_ANSWER_KEY),
        "combined_accuracy": round(100 * both / n, 1) if n else 0.0,  # Both scores equal to 1.
        "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
        "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
        "total_cost_usd": round(cost, 6),
        "cost_per_question_usd": round(cost / n, 6) if n else 0.0,
        "cost_per_correct_answer_usd": round(cost / correct, 6) if correct else None,
        "langsmith_cost_usd": langsmith_cost,
        "latency_p50_s": _round_or_none(percentile(latencies, 50), 2),
        "latency_p95_s": _round_or_none(percentile(latencies, 95), 2),
        "errors": sum(1 for row in rows if row.get("error")),
        "per_question": sorted(rows, key=lambda row: row.get("id") or ""),
    }


def row_from_result(result: Any, ids_by_question: dict[str, str]) -> dict:
    """One row per example from a LangSmith ExperimentResultRow."""
    run, example = result["run"], result["example"]
    outputs = run.outputs or {}
    metrics = outputs.get("metrics") or {}
    scores = {r.key: r.score for r in result["evaluation_results"]["results"]}
    question = (example.inputs or {}).get(QUESTION_KEY, "")
    latency = metrics.get("latency_s")
    if latency is None and run.end_time and run.start_time:
        latency = (run.end_time - run.start_time).total_seconds()
    return {
        "id": ids_by_question.get(question),
        "question": question,
        "tools_used": outputs.get("tools_used"),
        "answer": outputs.get("answer"),
        TOOL_USAGE_KEY: scores.get(TOOL_USAGE_KEY),
        CORRECT_ANSWER_KEY: scores.get(CORRECT_ANSWER_KEY),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "cost_usd": metrics.get("cost_usd"),
        "latency_s": latency,
        "error": run.error,
    }


def _fmt(value: Any, template: str = "{}") -> str:
    return "n/a" if value is None else template.format(value)


def print_summary(summary: dict) -> None:
    print(
        f"\nExperiment {summary['experiment'] or summary['prefix']} · "
        f"AGENT_CONFIG={summary['agent_config']} · {summary['model']}"
    )
    print(f"  Tool usage:            {summary['tool_usage_accuracy']} %")
    print(f"  Correct answer:        {summary['answer_accuracy']} %")
    print(f"  Both correct:          {summary['combined_accuracy']} %")
    print(f"  Input / output tokens: {summary['input_tokens']:,} / {summary['output_tokens']:,}")
    print(
        f"  Total cost (ours):     USD {summary['total_cost_usd']:.6f} · "
        f"per question USD {summary['cost_per_question_usd']:.6f}"
    )
    print(f"  Total cost (LangSmith): {_fmt(summary['langsmith_cost_usd'], 'USD {:.6f}')}")
    print(f"  Latency p50 / p95:     {_fmt(summary['latency_p50_s'])} s / {_fmt(summary['latency_p95_s'])} s")
    for row in summary["per_question"]:
        if row[TOOL_USAGE_KEY] != 1 or row[CORRECT_ANSWER_KEY] != 1 or row["error"]:
            error = f" error={str(row['error'])[:80]}" if row["error"] else ""
            print(
                f"  ✗ {row['id']}: tools={row['tools_used']} tool_usage={row[TOOL_USAGE_KEY]} "
                f"correct={row[CORRECT_ANSWER_KEY]}{error}"
            )


def experiment_path(prefix: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"qa_experiment_{prefix}.json"


def save_summary(summary: dict, results_dir: Path = RESULTS_DIR) -> Path:
    path = experiment_path(summary["prefix"], results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def ensure_dataset(client: Any, cases: list[dict]) -> None:
    """Upload the reference set to LangSmith the first time."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(DATASET_NAME, description="Workshop 1 reference questions (SECOP sample)")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {QUESTION_KEY: case["pregunta"]},
                "outputs": {
                    EXPECTED_ANSWER_KEY: case["respuesta_esperada"],
                    EXPECTED_TOOLS_KEY: case["herramientas_esperadas"],
                },
            }
            for case in cases
        ],
    )
    logger.info("Created LangSmith dataset %s with %d examples", DATASET_NAME, len(cases))


def langsmith_total_cost(client: Any, experiment: str) -> float | None:
    """`total_cost` of the experiment's project; None when LangSmith does not know the model's rate."""
    try:
        client.flush()
        project = client.read_project(project_name=experiment, include_stats=True)
    except Exception as exc:  # noqa: BLE001  (the statistics are optional)
        logger.warning("Could not read the LangSmith cost of %s: %s", experiment, exc)
        return None
    return float(project.total_cost) if project.total_cost is not None else None


def _reset_langsmith_env_cache() -> None:
    """langsmith caches environment variables; drop the cache after `load_environment` edits them."""
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except (ImportError, AttributeError):
        pass


def run_experiment(prefix: str, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> dict:
    """Evaluate the configured agent on the reference dataset and return the summary."""
    from langsmith import Client

    cases = load_reference_cases()
    client = Client()
    ensure_dataset(client, cases)
    config, model = qa_agent.normalize_config(), model_from_env()
    results = client.evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[tool_usage, correct_answer, cost_usd],
        experiment_prefix=prefix,
        max_concurrency=max_concurrency,
        description=f"AGENT_CONFIG={config} · {qa_agent.CONFIG_DESCRIPTIONS[config]}",
        metadata={
            "agent_config": config,
            "model": model,
            "judge_model": judge_model(),
            "data_mode": data_mode(),
            "team_name": team_name(),
        },
    )
    ids_by_question = {case["pregunta"]: case["id"] for case in cases}
    rows = [row_from_result(result, ids_by_question) for result in results]
    return summarize_experiment(
        rows,
        prefix,
        config,
        model,
        results.experiment_name,
        langsmith_total_cost(client, results.experiment_name),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Q&A agent on the reference set in LangSmith.")
    parser.add_argument("prefix", nargs="?", help="Experiment prefix (default: config-<AGENT_CONFIG>).")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    args = parser.parse_args(argv)
    load_environment()
    _reset_langsmith_env_cache()
    logging.basicConfig(level=logging.WARNING)
    missing = [name for name in REQUIRED_KEYS if not os.getenv(name)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)} (see .env.example).", file=sys.stderr)
        return 2
    if data_mode() != SAMPLE_MODE:
        print(f"Note: the reference answers assume {DATA_MODE_ENV}={SAMPLE_MODE}.", file=sys.stderr)
    summary = run_experiment(args.prefix or f"config-{qa_agent.normalize_config()}", args.max_concurrency)
    print_summary(summary)
    print(f"\nSaved to {save_summary(summary)}")
    print("Review the experiment in LangSmith > Datasets & Experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
