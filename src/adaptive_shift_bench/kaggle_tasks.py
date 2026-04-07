from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_shift_bench.engine import run_scenario
from adaptive_shift_bench.reporting import aggregate_attempts, write_report_bundle
from adaptive_shift_bench.scenarios import DEFAULT_ATTEMPTS, build_core_suite, get_scenario


class KBenchAdapter:
    def __init__(self, llm: Any):
        self.llm = llm

    def reset(self) -> None:
        return None

    def prompt(self, message: str) -> str:
        return self.llm.prompt(message)


def _report_attempt_path(scenario_id: str, attempt_index: int, output_dir: str | Path = "reports") -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{scenario_id}-attempt-{attempt_index + 1}.json"


def _write_attempt_report(report: dict[str, object], scenario_id: str, attempt_index: int) -> Path:
    path = _report_attempt_path(scenario_id, attempt_index)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_kbench_tasks():
    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kaggle_benchmarks is required to build Kaggle tasks. "
            "Import this function inside a Kaggle notebook or an environment with kaggle_benchmarks installed."
        ) from exc

    @kbench.task(name="adaptive_shift_attempt")
    def adaptive_shift_attempt(llm, scenario_id: str, attempt_index: int = 0) -> float:
        scenario = get_scenario(scenario_id)
        result = run_scenario(KBenchAdapter(llm), scenario, attempt_index=attempt_index)
        _write_attempt_report(
            {
                "scenario_id": result.scenario_id,
                "family": result.family.value,
                "attempt_index": result.attempt_index,
                "passed": result.passed,
                "score": result.score,
                "failure_tags": result.failure_tags,
                "selected_entities": result.selected_entities,
                "used_evidence_ids": result.used_evidence_ids,
                "trace_digest": result.trace_digest,
            },
            scenario_id,
            attempt_index,
        )
        return result.score

    @kbench.task(name="adaptive_shift_overall")
    def adaptive_shift_overall(llm) -> float:
        results = []
        for scenario in build_core_suite():
            for attempt_index in range(DEFAULT_ATTEMPTS):
                with kbench.chats.new(f"{scenario.id}-attempt-{attempt_index + 1}", orphan=True):
                    results.append(
                        run_scenario(
                            KBenchAdapter(llm),
                            scenario,
                            attempt_index=attempt_index,
                        )
                    )
        report = aggregate_attempts(results)
        write_report_bundle(report)
        return float(report["metrics"]["overall"])

    return adaptive_shift_attempt, adaptive_shift_overall
