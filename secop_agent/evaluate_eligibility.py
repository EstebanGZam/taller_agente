"""LangSmith eligibility dataset and deterministic evaluators, without an LLM judge."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

from langsmith import Client

from secop_agent import settings
from secop_agent.batch_eligibility import load_company
from secop_agent.eligibility_agent import EligibilityAgent
from secop_agent.tender_documents import normalize_model

DATASET = "secop-eligibility-key"


def ensure_dataset(client: Any) -> Any:
    """Create the twenty examples only if the dataset does not exist."""
    if client.has_dataset(dataset_name=DATASET):
        return client.read_dataset(dataset_name=DATASET)
    key = json.loads(settings.ELIGIBILITY_KEY_FILE.read_text(encoding="utf-8"))["processes"]
    dataset = client.create_dataset(
        DATASET, description=("20 SECOP tenders against the multipurpose company, compared in nominal pesos.")
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"file": process["file"]},
                "outputs": {
                    "verdict": process["expected_verdict"],
                    "requirements": process["expected_requirements"],
                },
            }
            for process in key
        ],
    )
    return dataset


def verdict_correct(run: Any, example: Any) -> dict:
    return {
        "key": "verdict_correct",
        "score": ((run.outputs or {}).get("verdict") == (example.outputs or {}).get("verdict")),
    }


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and set(actual) == set(expected)
    if isinstance(expected, bool) or expected is None:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
        )
    return actual == expected


def requirements_extraction(run: Any, example: Any) -> dict:
    expected = (example.outputs or {}).get("requirements", {})
    actual = (run.outputs or {}).get("requirements") or {}
    matches = sum(key in actual and _matches(actual[key], value) for key, value in expected.items())
    return {"key": "requirements_extraction", "score": matches / len(expected) if expected else 0}


def tool_usage(run: Any, example: Any) -> dict:
    used = set((run.outputs or {}).get("tools_used", []))
    return {"key": "tool_usage", "score": {"extract_requirements", "verify_requirements"}.issubset(used)}


def target(inputs: dict) -> dict:
    """Evaluate one repository PDF, adding extraction evidence to the agent's public result."""
    path = (settings.COURSE_DATA_DIR / inputs["file"]).resolve()
    if path.parent != settings.COURSE_DATA_DIR.resolve() or path.suffix.lower() != ".pdf":
        raise ValueError("The example must reference a PDF directly inside datos/.")
    agent = EligibilityAgent(path, load_company(), model=normalize_model())
    return {**agent.analyze(), "requirements": agent.requirements}


def main(argv: list[str] | None = None) -> int:
    settings.load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="?", default="eligibility-haiku")
    args = parser.parse_args(argv)
    for variable in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"):
        if not os.getenv(variable):
            parser.error(f"{variable} is missing.")
    client = Client()
    ensure_dataset(client)
    client.evaluate(
        target,
        data=DATASET,
        evaluators=[verdict_correct, requirements_extraction, tool_usage],
        experiment_prefix=args.prefix,
        max_concurrency=3,
        metadata={
            "model": normalize_model(),
            "company": "empresa_multiproposito.json",
            "strict": False,
            "team_name": settings.team_name(),
            "repository_url": settings.repository_url(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
