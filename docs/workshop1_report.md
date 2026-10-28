# Workshop 1 report · Tools and cost-quality

Team: `estebangzam` · Repository: <https://github.com/EstebanGZam/taller_agente> · Model: `anthropic:claude-haiku-5-5`
(USD 0.10 / 0.50 per million input / output tokens) · All runs on 2026-10-28.

## 1. Problem

A company wants to know which SECOP II procurement processes it should bid on. The workshop asks for an
agent with its own tools, traced in LangSmith, evaluated on a 15-20 item dataset, with two configurations
compared on accuracy, cost and latency (target: at least 85% accuracy at the lowest cost). The class notes
add: run the agent over the 20 tender PDFs in the repository, decide for each one whether the company
(`empresa_multiproposito.json`) should bid, expose it through a frontend, and record name, repository and
cost of those 20 analyses.

## 2. Two agents

**Q&A agent** (`secop_agent/qa_agent.py`). Answers questions about processes and the reference company with
three tools: `search_processes`, `get_company_profile`, `compute_financial_indicators`. Evaluated on the
course reference set (18 questions).

**Eligibility agent** (`secop_agent/eligibility_agent.py`). Reads one tender PDF and decides whether to bid,
with four tools: `extract_requirements`, `verify_requirements`, `get_company_profile`,
`compute_financial_indicators`. Evaluated on our key of 20 tenders (`data/eligibility_key.json`).

### Design decision: the model extracts, the code decides

Letting the model read the PDF and the company and give the verdict directly would be cheaper, but it would
not be auditable: threshold comparisons and percentage-of-budget arithmetic are exactly where language models
slip, and an instruction injected into a PDF could flip the answer. The design splits the work:

1. `extract_requirements` turns the PDF into a Pydantic `Requirements` object (thresholds, UNSPSC codes,
   modality, budget, direct-contracting and asset-sale flags). Missing thresholds are `null`, never guessed.
2. `select_similar_contracts` judges, contract by contract, whether each past contract of the company
   delivered the same kind of goods or services as the tender.
3. `verify_requirements` (`secop_agent/eligibility.py`) compares requirements and company with plain
   arithmetic and returns one check per requirement and a verdict: `ELIGIBLE`, `NOT_ELIGIBLE`,
   `NOT_APPLICABLE` (direct contracting or asset sale) or `MANUAL_REVIEW`.
4. The agent explains the result in Spanish and keeps the conversation for follow-up questions.

The verdict is therefore reproducible, every check is visible in the UI and in the JSON report, and
instructions injected in a PDF cannot change it.

## 3. Datasets

| Dataset | Items | Source | LangSmith name |
|---|---|---|---|
| Q&A reference set | 18 questions with expected answer and tools | Course (`datos/conjunto_referencia.json`) | `secop-taller1-referencia` |
| Eligibility key | 20 tenders with expected verdict and expected requirements | Built by hand from each PDF and the company (`data/eligibility_key.json`) | `secop-eligibility-key` |

Key distribution: 6 `ELIGIBLE` (03, 07, 08, 11, 14, 15), 2 `NOT_ELIGIBLE` (13: working capital and
experience; 18: no similar experience) and 12 `NOT_APPLICABLE` (10 direct contracts with a natural person,
12 inter-administrative, 17 asset sale). This matches the note in `empresa_multiproposito.json`: designed to
meet six processes and fail two.

## 4. Configuration comparison (Q&A agent)

| | A · course baseline | B · improved |
|---|---|---|
| Change | Original Spanish prompt and tool descriptions | Precise tool descriptions, compact tool output with summary and warnings, reference filter, exact modality and sorting, one prompt rule per question type |
| Correct answer (LLM judge) | 83.3 % (15/18) | **100.0 % (18/18)** |
| Correct tool usage | 100.0 % | 100.0 % |
| Input / output tokens | 80,048 / 9,904 | 97,411 / 5,208 |
| Total cost | USD 0.012957 | **USD 0.012345** |
| Cost per correct answer | USD 0.000864 | **USD 0.000686** |
| Latency p50 / p95 | 4.29 s / 20.22 s | **2.73 s / 4.52 s** |

B is the only configuration above 85 % and is also cheaper and faster: it reads more input (richer tool
descriptions) but writes half as many output tokens, which cost five times more. A fails the three questions
that name a process by its reference (R13 `LP-DT-CAU-003-2019`, R14 `SA-004-2023`, R16
`FLO-INFRA-SA-MC-007-2025`): the original `search_processes` has no reference filter, so the agent searched
by keywords, found nothing and answered that the process does not exist. Experiments: `A-baseline-5095108d`,
`B-improved-913926ac`.

## 5. The 20 tenders (eligibility agent)

`python -m secop_agent.batch_eligibility --company empresa_multiproposito.json` →
`results/eligibility_processes.{csv,md,json}`.

| Metric | Value |
|---|---|
| Accuracy against the key | **20 / 20 (100 %)** |
| Verdicts | 6 ELIGIBLE · 2 NOT_ELIGIBLE · 12 NOT_APPLICABLE |
| Total cost of the 20 analyses | **USD 0.042274** |
| Mean cost per tender | USD 0.002114 |
| Tokens | 206,783 input · 43,192 output |
| Latency p50 | 12.8 s per tender (3 in parallel, about 1.6 min for the batch) |

LangSmith evaluation (`eligibility-haiku-v3-537350bb`, the same 20 tenders as the dataset
`secop-eligibility-key`, an independent second run): verdict correct **1.000**, requirements extraction
0.954 (share of expected requirement fields extracted correctly), tool usage 1.000, 0 errors, USD 0.043863,
p50 12.7 s.

### How we reached 20/20

| Run | Accuracy | Error | Fix |
|---|---|---|---|
| v1 | 19/20 | 20 (special-regime contract for a person) read as a competitive call → `MANUAL_REVIEW` | Define direct contracting by the absence of a call for offers (no offer-submission date, no selection criterion), whatever legal ground it cites |
| v2 | 19/20 | 20 fixed, but 18 became a false `ELIGIBLE`: asked for "the IDs of similar contracts", the model returned every contract | One structured judgment per contract (`same_nature` + reason); repeated probes: 30/30 correct on 08 and 18 |
| v3 | **20/20** | — | — |

## 6. Cost and tracing

- Every model call goes through `settings.chat_model()`; a `CostTracker` callback (`secop_agent/costs.py`)
  sums tokens and USD with Anthropic's public rates, including the calls made inside tools. LangSmith gets
  the same traces, with tokens, cost and latency per run.
- Our cost and LangSmith's agree closely (A: identical, USD 0.012957; B: USD 0.012345 vs. 0.012558, a 1.7 %
  gap).
- One full run of the 20 tenders costs about USD 0.04; one Q&A experiment about USD 0.013.

## 7. Frontend

`python app.py` (or the last notebook cell, with a public Gradio link) opens *SECOP | Bid eligibility*:

1. **Inputs**: an optional company JSON (default: `empresa_multiproposito.json`), the model (Haiku 5.5 or
   Sonnet 5.5) and the strict SMMLV switch.
2. **Analyze the 20 repository tenders**, or upload other tender PDFs: a table with verdict, summary, cost,
   tokens and latency per tender, the team/repository/cost summary, and a CSV download.
3. **Review a process**: every check of the selected tender with the agent's explanation, and a chat for
   follow-up questions answered by the same agent with its memory.

## 8. Limitations

- The key compares amounts in nominal pesos. With a strict per-year SMMLV conversion (`--strict`), tender
  11 (950M vs. 900M) would be `NOT_ELIGIBLE`; the choice is documented, not hidden.
- Scanned PDFs without selectable text are rejected (no OCR).
- Similarity of experience is a model judgment; it is validated on this dataset (30/30 probes) but is the
  least deterministic step.
- The 20-tender key was built by us; a second reviewer would make it stronger.
- Prompt-injection defence is basic (delimiters, explicit rule, deterministic verdict); workshop 4 covers it.
