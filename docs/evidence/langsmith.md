# LangSmith evidence

Generated from the LangSmith API on 2026-10-28 (workspace of the team `estebangzam`). The links open only
for members of the workspace; screenshots of the same views are in this folder when available.

## Experiments (Datasets & Experiments)

| Experiment | What | Runs | Errors | Feedback (mean) | Tokens in / out | LangSmith cost | p50 latency |
|---|---|---|---|---|---|---|---|
| `A-baseline-f20d7ff7` | Q&A, configuration A (course baseline) | 18 | 0 | correct_answer 0.83, cost_usd 0.000678, tool_usage 1.00 | 77,684 / 8,968 | USD 0.012252 | 4.05 s |
| `B-improved-4e8aed68` | Q&A, configuration B (improved) | 18 | 0 | correct_answer 1.00, cost_usd 0.000694, tool_usage 1.00 | 97,358 / 5,319 | USD 0.012395 | 3.01 s |
| `eligibility-haiku-16b096ee` | Eligibility, the 20 tenders | 20 | 0 | requirements_extraction 0.95, tool_usage 1.00, verdict_correct 1.00 | 212,655 / 43,471 | USD 0.043001 | 12.01 s |

## Tracing project `secop-taller1-estebangzam`

190 root traces. By entry point:

| Root run | Traces |
|---|---|
| `eligibility_analyze` | 124 |
| `select_similar_contracts` | 50 |
| `extract_requirements` | 9 |
| `eligibility_follow_up` | 2 |
| `secop-qa-agent-B` | 2 |
| `RunnableSequence` | 2 |
| `smoke-test` | 1 |

## One full trace: tender 13 analysed from the frontend

Trace `01a18452-2928-76cc-b64f-9b066f2feb17` · verdict `NOT_ELIGIBLE` · 14.0 s · https://smith.langchain.com/o/be862a28-1dbf-4dec-b6c4-f58119a8f835/projects/p/32798b00-b8b7-4974-9be1-01a45287130d/r/01a18452-2928-76cc-b64f-9b066f2feb17?poll=true

The agent decides which tools to call; each model call records its tokens and latency.

```
chain  eligibility_analyze                 14.00 s
  chain  analyze_tender                      13.60 s
    chain  model                                1.59 s
      llm    ChatAnthropic                        1.59 s  tokens 889/132
    chain  tools                                5.87 s
      tool   extract_requirements                 5.87 s
      chain  extract_requirements                 5.87 s
        chain  RunnableSequence                     5.87 s
          llm    ChatAnthropic                        5.86 s  tokens 4,018/1,519
          parser PydanticToolsParser                  0.00 s
    chain  tools                                5.87 s
      tool   get_company_profile                  5.87 s
    chain  model                                0.90 s
      llm    ChatAnthropic                        0.89 s  tokens 3,701/29
    chain  tools                                0.00 s
      tool   verify_requirements                  0.00 s
    chain  model                                5.23 s
      llm    ChatAnthropic                        5.23 s  tokens 5,123/1,035
```

Model calls in the trace: 4, tokens 13,731 in / 2,715 out.
