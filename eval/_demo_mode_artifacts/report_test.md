# CareBridge test evaluation

> Seed questions are machine-drafted, especially Telugu/Hindi/code-mixed items; have fluent speakers review them before reporting these results.

Judge mode: `offline_lexical_heuristic`.

| Metric | Gate enabled | No-gate baseline |
|---|---:|---:|
| Answer accuracy | 0.600 | 0.600 |
| Citation correctness | 1.000 | 1.000 |
| Red-flag recall | 1.000 | 1.000 |
| Red-flag false-alarm rate | 0.000 | 0.000 |
| Abstention precision | 0.556 | 0.556 |
| Abstention recall | 1.000 | 1.000 |
| Answered coverage | 0.300 | 0.300 |
| ECE | 0.491 | 0.491 |
| Mean latency (ms) | 12.020 | 6.013 |
| Throughput (questions/sec) | 83.195 | 166.304 |

## Per-language gated metrics

| Language | Red-flag recall | False-alarm rate | Answer accuracy |
|---|---:|---:|---:|
| en | 1.000 | 0.000 | 1.000 |
| hi | 1.000 | 0.000 | 0.000 |
| te | 1.000 | 0.000 | 0.000 |

Accuracy uses the configured LLM judge when an endpoint is available; otherwise the report labels its local lexical/semantic heuristic. Review the CSV before making safety or efficacy claims.
