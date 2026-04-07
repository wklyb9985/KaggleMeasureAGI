# Adaptive Shift Bench

`adaptive-shift-bench` is an offline synthetic benchmark for measuring whether models can adapt to interface changes in real software ecosystems after seeing new local evidence.

It is built to support:

- real package, SDK, and model names with fictional future changes
- deterministic offline docs, registries, and execution feedback
- short multi-turn adaptation episodes plus optional long-loop stress cases
- `pass@1`, `pass@5`, `avg5`, and weighted overall scoring
- import into Kaggle notebooks with `kaggle-benchmarks`

## Layout

- `src/adaptive_shift_bench/`: benchmark package
- `tests/`: local unit tests using scripted adapters

## Local test run

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Kaggle usage

In a Kaggle notebook with `kaggle_benchmarks` available:

```python
from adaptive_shift_bench.kaggle_tasks import build_kbench_tasks

adaptive_shift_attempt, adaptive_shift_overall = build_kbench_tasks()

# Run the leaderboard task
adaptive_shift_overall.run(llm=kbench.llm)
```

