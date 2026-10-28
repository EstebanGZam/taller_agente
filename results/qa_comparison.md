# Q&A agent configurations · Workshop 1

Reference set (18 questions, `DATA_MODE=sample`). Our cost comes from `secop_agent.costs` (Anthropic's public rates); LangSmith may not know the model's rate.

| | A-baseline | B-improved |
|---|---|---|
| LangSmith experiment | `A-baseline-f20d7ff7` | `B-improved-4e8aed68` |
| Configuration | A | B |
| Model | `anthropic:claude-haiku-5-5` | `anthropic:claude-haiku-5-5` |
| Change | Course's original prompt and tool descriptions (baseline) | Precise descriptions, compact output with summary and warnings, reference filter, exact modality and sorting; prompt with one rule per question type |
| Questions | 18 | 18 |
| Tool usage | 100.0 % | 100.0 % |
| Correct answer | 83.3 % | 100.0 % |
| Both correct | 83.3 % | 100.0 % |
| Input / output tokens | 77,684 / 8,968 | 97,358 / 5,319 |
| Total cost (ours) | USD 0.012252 | USD 0.012395 |
| Cost per question | USD 0.000681 | USD 0.000689 |
| Cost per correct answer | USD 0.000817 | USD 0.000689 |
| Total cost (LangSmith) | USD 0.012154 | USD 0.012780 |
| Latency p50 | 4.05 s | 3.01 s |
| Latency p95 | 6.32 s | 3.69 s |

**Verdict:** Meeting the 85% correct-answer criterion: B-improved (100.0%). Cheapest per correct answer: B-improved (USD 0.000689). Recommended: B-improved (>= 85% at the lowest cost per question).
