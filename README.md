# Adaptive Shift Bench

`adaptive-shift-bench` is an offline synthetic benchmark for measuring whether models can adapt to interface changes in real software ecosystems after seeing new local evidence.

It is built to support:

- real package, SDK, and model names with fictional future changes
- deterministic offline docs, registries, and execution feedback
- short multi-turn adaptation episodes plus optional long-loop stress cases
- `pass@1`, `pass@5`, `avg5`, and weighted overall scoring
- import into Kaggle notebooks with `kaggle-benchmarks`

## V2 Status

- New V2 sequence reports are emitted as benchmark version `2.1`.
- Any older V2 report that does not include `benchmark_metadata.benchmark_version` should be treated as `obsolete_pre_fix_v2` and not compared directly with `2.1` results.
- The canonical learning suite is now the expanded `v2_learning` variant `b_expanded`, emitted as benchmark version `2.4-learning-b-expanded`.
- Older learning reports should not be compared directly with `2.4-learning-b-expanded`.
- The new strict prior-proof learning suites are emitted as benchmark version `3.0-strict-prior-proof`.
- `v3_learning_strict_standard` is the direct-score public strict suite.
- `v3_learning_strict_hard` is the matched hard variant for score-shift analysis.

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

## Local Kaggle-style smoke runs

You can exercise the same task builders locally by installing the bundled shim at runtime:

```python
from adaptive_shift_bench.kaggle_tasks import build_kbench_v2_learning_tasks
from adaptive_shift_bench.local_kaggle_mock import LocalTaskLLM, patched_local_kaggle_benchmarks
from adaptive_shift_bench.llm import ScriptedLLMAdapter

with patched_local_kaggle_benchmarks():
    _, adaptive_shift_v2_learning_sequence, _ = build_kbench_v2_learning_tasks(output_dir="workspace/local_kbench_demo")
    llm = LocalTaskLLM(lambda session_key: ScriptedLLMAdapter([...]))
    adaptive_shift_v2_learning_sequence.run(
        llm=llm,
        sequence_id="v2-learning-openai-revision",
        attempt_index=0,
    )
```

Or use the local runner:

```bash
PYTHONPATH=src python -m adaptive_shift_bench.local_kaggle_runner \
  --backend codex \
  --suite v2_learning \
  --task sequence \
  --model gpt-5.4-mini \
  --sequence-id v2-learning-openai-revision
```

For the new strict suites:

```bash
PYTHONPATH=src python -m adaptive_shift_bench.local_kaggle_runner \
  --backend codex \
  --suite v3_learning_strict_standard \
  --task overall \
  --model gpt-5.4-mini
```

The Kaggle notebook entrypoints for the public strict suites are:

- `notebooks/adaptive_shift_strict_standard_kbench.py`
- `notebooks/adaptive_shift_strict_hard_kbench.py`
