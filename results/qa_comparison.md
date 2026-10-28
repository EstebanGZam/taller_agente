# Q&A agent configurations · Workshop 1

Reference set (18 questions, `DATA_MODE=sample`). Our cost comes from `secop_agent.costs` (Anthropic's public rates); LangSmith may not know the model's rate.

| | A-baseline | B-improved |
|---|---|---|
| LangSmith experiment | `A-baseline-5095108d` | `B-improved-913926ac` |
| Configuration | A | B |
| Model | `anthropic:claude-haiku-5-5` | `anthropic:claude-haiku-5-5` |
| Change | Course's original prompt and tool descriptions (baseline) | Precise descriptions, compact output with summary and warnings, reference filter, exact modality and sorting; prompt with one rule per question type |
| Questions | 18 | 18 |
| Tool usage | 100.0 % | 100.0 % |
| Correct answer | 83.3 % | 100.0 % |
| Both correct | 83.3 % | 100.0 % |
| Input / output tokens | 80,048 / 9,904 | 97,411 / 5,208 |
| Total cost (ours) | USD 0.012957 | USD 0.012345 |
| Cost per question | USD 0.000720 | USD 0.000686 |
| Cost per correct answer | USD 0.000864 | USD 0.000686 |
| Total cost (LangSmith) | USD 0.012957 | USD 0.012558 |
| Latency p50 | 4.29 s | 2.73 s |
| Latency p95 | 20.22 s | 4.52 s |

**Verdict:** Meeting the 85% correct-answer criterion: B-improved (100.0%). Cheapest per correct answer: B-improved (USD 0.000686). Recommended: B-improved (>= 85% at the lowest cost per question).
