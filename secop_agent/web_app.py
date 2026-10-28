"""Gradio interface for tender eligibility analysis and per-process follow-up."""

from __future__ import annotations

import os
import tempfile
from html import escape
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr

from secop_agent import settings
from secop_agent.batch_eligibility import analyze_batch, load_company, write_results

COLORS = {
    "ELIGIBLE": "#167042",
    "NOT_ELIGIBLE": "#b02f36",
    "NOT_APPLICABLE": "#516579",
    "MANUAL_REVIEW": "#926300",
}
EMOJIS = {"ELIGIBLE": "✅", "NOT_ELIGIBLE": "❌", "NOT_APPLICABLE": "➖", "MANUAL_REVIEW": "⚠️"}
PROGRESS = gr.Progress()
CHECK_ICONS = {True: "✅", False: "❌", None: "⚠️"}
CSS = """
.gradio-container {max-width: 1440px !important; font-family: Arial, sans-serif;}
#intro {border-left: 6px solid #20547a; padding: 8px 20px; background: #edf4f8;}
#intro h1 {font-size: 30px; color: #173b57;}
#intro p {max-width: 76ch; color: #344d61;}
.requirements-table {width: 100%; border-collapse: collapse; margin: 16px 0;}
.requirements-table th, .requirements-table td {padding: 10px; text-align: left;
 border-bottom: 1px solid #cbd7df; overflow-wrap: anywhere;}
.requirements-table th {background: #edf4f8; color: #173b57;}
.process-detail {padding: 12px 18px; border-left: 5px solid #20547a;}
.process-detail pre {white-space: pre-wrap; font-family: inherit;}
"""


def _file_path(file: object) -> Path:
    return Path(file if isinstance(file, (str, Path)) else file.name)


def detail_html(result: dict) -> str:
    """Escape every external field and allow only HTTPS LangSmith trace links."""

    def escaped(value: object) -> str:
        return escape(str(value if value is not None else "—"), quote=True)

    verdict = result["verdict"]
    color = COLORS.get(verdict, "#516579")
    rows = "".join(
        f"<tr><td>{escaped(check['requirement'])}</td><td>{escaped(check['required'])}</td>"
        f"<td>{escaped(check['company_value'])}</td><td>{CHECK_ICONS.get(check['passed'], '⚠️')} "
        f"{escaped(check.get('detail', ''))}</td></tr>"
        for check in result["checks"]
    )
    table = (
        "<table class='requirements-table'><thead><tr><th>Requirement</th><th>Required</th>"
        "<th>Company</th><th>Passed / detail</th></tr></thead><tbody>" + rows + "</tbody></table>"
        if rows
        else ""
    )
    warnings = "".join("<li>" + escaped(warning) + "</li>" for warning in result.get("warnings", []))
    link = ""
    url = result.get("trace_url")
    if url and urlparse(url).scheme == "https" and urlparse(url).hostname == "smith.langchain.com":
        link = (
            f"<p><a href='{escaped(url)}' target='_blank' rel='noopener noreferrer'>"
            "View LangSmith trace</a></p>"
        )
    elif result.get("run_id"):
        link = "<p>Trace identifier: <code>" + escaped(result["run_id"]) + "</code></p>"
    return (
        f"<section class='process-detail' style='border-color:{color}'>"
        f"<h2 style='color:{color}'>{EMOJIS.get(verdict, '⚠️')} {escaped(verdict)}</h2>"
        f"<p><strong>{escaped(result.get('reference'))}</strong> | {escaped(result.get('entity'))}</p>"
        f"<p>{escaped(result['summary'])}</p>{table}"
        f"{'<ul>' + warnings + '</ul>' if warnings else ''}"
        f"<pre>{escaped(result.get('explanation', ''))}</pre>{link}</section>"
    )


def analyze(
    pdfs: list | None = None,
    company_json: object = None,
    model: str = "claude-haiku-5-5",
    strict: bool = False,
    progress: gr.Progress | None = PROGRESS,
) -> tuple:
    """Analyze uploaded PDFs, or the twenty repository tenders when none are uploaded."""
    settings.load_environment()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise gr.Error("ANTHROPIC_API_KEY is missing. Set it in the environment or .env before analyzing.")
    try:
        company = load_company(
            _file_path(company_json) if company_json else settings.MULTIPURPOSE_COMPANY_FILE
        )
        paths = [_file_path(pdf) for pdf in pdfs] if pdfs else list(settings.COURSE_DATA_DIR.glob("*.pdf"))
        if progress is not None:
            progress(0, desc="Analyzing tenders…")
        batch = analyze_batch(paths, company, model, workers=3, strict=bool(strict))
    except (ValueError, OSError) as exc:
        raise gr.Error("Could not start analysis: " + escape(str(exc))) from exc
    folder = Path(tempfile.mkdtemp(prefix="eligibility-"))
    download = str(write_results(batch, folder))
    results = {str(result["n"]): result for result in batch["results"]}
    histories = {
        key: [{"role": "assistant", "content": escape(result["explanation"])}]
        if result["explanation"]
        else []
        for key, result in results.items()
    }
    rows = [
        [
            result["n"],
            escape(result["file"]),
            escape(result.get("reference") or ""),
            escape(result.get("entity") or ""),
            EMOJIS[result["verdict"]] + " " + result["verdict"],
            escape(result["summary"]),
            result["metrics"]["cost_usd"],
            result["metrics"]["input_tokens"] + result["metrics"]["output_tokens"],
            result["metrics"]["latency_s"],
        ]
        for result in batch["results"]
    ]
    summary = batch["summary"]
    counts = " | ".join(
        f"{EMOJIS[verdict]} {verdict}: {summary['counts'].get(verdict, 0)}" for verdict in EMOJIS
    )
    text = (
        f"{len(rows)} unique tenders · {counts}\n\nTotal analysis cost: USD {summary['total_cost_usd']:.6f}"
    )
    if not summary["cost_complete"]:
        text += " (some costs are undetermined)"
    if summary["evaluated"]:
        text += f" · Correct: {summary['correct']}/{summary['evaluated']} ({summary['accuracy_pct']:.2f}%)"
        if strict:
            text += " · The key uses nominal pesos; strict SMMLV conversion may change verdicts."
    choices = [(f"{result['n']}. {escape(result['file'])}", str(result["n"])) for result in batch["results"]]
    first = choices[0][1]
    if progress is not None:
        progress(1, desc="Analysis complete")
    return (
        batch["agents"],
        results,
        histories,
        text,
        rows,
        download,
        gr.update(choices=choices, value=first),
        detail_html(results[first]),
        histories[first],
    )


def analyze_repository(
    company_json: object, model: str, strict: bool, progress: gr.Progress | None = PROGRESS
) -> tuple:
    return analyze(None, company_json, model, strict, progress)


def view_detail(number: str, results: dict, histories: dict) -> tuple[str, list]:
    if number not in (results or {}):
        return "", []
    return detail_html(results[number]), histories.get(number, [])


def chat(message: str, number: str, agents: dict, histories: dict) -> tuple[str, list, dict]:
    if not message or not message.strip():
        return "", histories.get(number, []), histories
    agent = (agents or {}).get(number)
    if agent is None:
        raise gr.Error("Select a process with a completed analysis.")
    try:
        answer = agent.ask(message)
    except Exception as exc:
        raise gr.Error("Could not answer. Check the connection and model configuration.") from exc
    history = list(histories.get(number, [])) + [
        {"role": "user", "content": escape(message)},
        {"role": "assistant", "content": escape(answer)},
    ]
    return "", history, {**histories, number: history}


def build_demo() -> gr.Blocks:
    """Build the UI without starting a server or constructing model clients."""
    with gr.Blocks(
        title="SECOP | Bid eligibility", css=CSS, theme=gr.themes.Base(font=["Arial", "sans-serif"])
    ) as demo:
        gr.Markdown(
            "# Which tenders can we bid on?\n"
            "Analyze the 20 repository tenders with the multipurpose company, "
            "or upload other tender PDFs and your company profile. Agent explanations are in Spanish.",
            elem_id="intro",
        )
        agents, results, histories = gr.State({}), gr.State({}), gr.State({})
        with gr.Row():
            company = gr.File(label="Company JSON (optional)", file_types=[".json"], type="filepath")
            with gr.Column():
                model = gr.Dropdown(
                    ["claude-haiku-5-5", "claude-sonnet-5-5"], value="claude-haiku-5-5", label="Model"
                )
                strict = gr.Checkbox(label="Convert experience to each year's SMMLV", value=False)
        repository = gr.Button("Analyze the 20 repository tenders", variant="primary")
        with gr.Accordion("Analyze other PDFs", open=False):
            pdfs = gr.File(label="Tender PDFs", file_types=[".pdf"], file_count="multiple", type="filepath")
            upload = gr.Button("Analyze uploaded PDFs")
        summary = gr.Markdown("Results will appear here after analysis.")
        table = gr.Dataframe(
            headers=[
                "#",
                "File",
                "Reference",
                "Entity",
                "Verdict",
                "Summary",
                "Cost USD",
                "Tokens",
                "Latency s",
            ],
            datatype=["number", "str", "str", "str", "str", "str", "number", "number", "number"],
            interactive=False,
            wrap=True,
        )
        download = gr.File(label="Download results as CSV")
        gr.Markdown("## Review a process")
        selected = gr.Dropdown(choices=[], label="Process", interactive=True)
        detail = gr.HTML()
        conversation = gr.Chatbot(
            label="Questions about this process", type="messages", sanitize_html=True, height=260
        )
        question = gr.Textbox(label="Question", placeholder="Which requirement do we still need to meet?")
        outputs = [agents, results, histories, summary, table, download, selected, detail, conversation]
        repository.click(
            analyze_repository,
            [company, model, strict],
            outputs,
            concurrency_limit=1,
            concurrency_id="analysis",
        )
        upload.click(
            analyze, [pdfs, company, model, strict], outputs, concurrency_limit=1, concurrency_id="analysis"
        )
        selected.change(view_detail, [selected, results, histories], [detail, conversation])
        question.submit(
            chat,
            [question, selected, agents, histories],
            [question, conversation, histories],
            concurrency_limit=1,
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    settings.load_environment()
    build_demo().queue().launch(share="COLAB_RELEASE_TAG" in os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
