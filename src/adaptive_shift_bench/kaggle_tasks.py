import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from adaptive_shift_bench.engine import run_scenario, run_sequence
from adaptive_shift_bench.reporting import (
    aggregate_attempts,
    aggregate_sequence_results,
    write_report_bundle,
    write_sequence_report_bundle,
)
from adaptive_shift_bench.scenarios import (
    DEFAULT_ATTEMPTS,
    build_core_suite,
    build_v2_learning_sequences,
    build_v2_sequences,
    get_scenario,
    get_v2_learning_scenario,
    get_v2_learning_sequence,
    get_v2_scenario,
    get_v2_sequence,
)

_RUN_NAMESPACE = os.environ.get("ADAPTIVE_SHIFT_RUN_ID", f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}")


class KBenchAdapter:
    def __init__(self, llm: Any):
        self.llm = llm

    def reset(self) -> None:
        return None

    def prompt(self, message: str) -> str:
        return self.llm.prompt(message)


def _resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    base = Path(os.environ.get("ADAPTIVE_SHIFT_OUTPUT_DIR", "reports"))
    return base / _RUN_NAMESPACE


def _report_attempt_path(
    name: str,
    attempt_index: int,
    *,
    prefix: str = "attempt",
    output_dir: str | Path | None = None,
) -> Path:
    base = _resolve_output_dir(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"
    return base / f"{prefix}-{name}-attempt-{attempt_index + 1}-{suffix}.json"


def _write_attempt_report(
    report: dict[str, object],
    name: str,
    attempt_index: int,
    *,
    prefix: str = "attempt",
    output_dir: str | Path | None = None,
) -> Path:
    path = _report_attempt_path(name, attempt_index, prefix=prefix, output_dir=output_dir)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_kbench_tasks(output_dir: str | Path | None = None):
    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kaggle_benchmarks is required to build Kaggle tasks. "
            "Import this function inside a Kaggle notebook or an environment with kaggle_benchmarks installed."
        ) from exc
    resolved_output_dir = _resolve_output_dir(output_dir)

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
            prefix="scenario",
            output_dir=resolved_output_dir,
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
        write_report_bundle(report, resolved_output_dir)
        return float(report["metrics"]["overall"])

    return adaptive_shift_attempt, adaptive_shift_overall


def build_kbench_v2_tasks(output_dir: str | Path | None = None):
    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kaggle_benchmarks is required to build Kaggle tasks. "
            "Import this function inside a Kaggle notebook or an environment with kaggle_benchmarks installed."
        ) from exc
    resolved_output_dir = _resolve_output_dir(output_dir) / "v2"

    @kbench.task(name="adaptive_shift_v2_attempt")
    def adaptive_shift_v2_attempt(llm, scenario_id: str, attempt_index: int = 0) -> float:
        scenario = get_v2_scenario(scenario_id)
        result = run_scenario(
            KBenchAdapter(llm),
            scenario,
            attempt_index=attempt_index,
            prompt_style="release_note",
        )
        _write_attempt_report(
            {
                "scenario_id": result.scenario_id,
                "sequence_id": result.sequence_id,
                "sequence_stage": result.sequence_stage.value if result.sequence_stage is not None else None,
                "family": result.family.value,
                "attempt_index": result.attempt_index,
                "passed": result.passed,
                "score": result.score,
                "failure_tags": result.failure_tags,
                "selected_entities": result.selected_entities,
                "used_evidence_ids": result.used_evidence_ids,
                "surfaced_evidence_ids": result.surfaced_evidence_ids,
                "score_breakdown": result.score_breakdown,
                "trace_digest": result.trace_digest,
            },
            scenario_id,
            attempt_index,
            prefix="v2-scenario",
            output_dir=resolved_output_dir,
        )
        return result.score

    @kbench.task(name="adaptive_shift_v2_sequence")
    def adaptive_shift_v2_sequence(llm, sequence_id: str, attempt_index: int = 0) -> float:
        sequence = get_v2_sequence(sequence_id)
        with kbench.chats.new(f"{sequence.id}-attempt-{attempt_index + 1}", orphan=True):
            result = run_sequence(
                KBenchAdapter(llm),
                sequence,
                attempt_index=attempt_index,
                prompt_style="release_note",
            )
        _write_attempt_report(
            {
                "sequence_id": result.sequence_id,
                "family": result.family.value,
                "attempt_index": result.attempt_index,
                "passed": result.passed,
                "overall_score": result.overall_score,
                "score_breakdown": result.score_breakdown,
                "learned_rules": result.learned_rules,
                "stage_results": [
                    {
                        "scenario_id": stage.scenario_id,
                        "sequence_stage": stage.sequence_stage.value if stage.sequence_stage is not None else None,
                        "score": stage.score,
                        "passed": stage.passed,
                        "score_breakdown": stage.score_breakdown,
                        "used_evidence_ids": stage.used_evidence_ids,
                        "surfaced_evidence_ids": stage.surfaced_evidence_ids,
                    }
                    for stage in result.stage_results
                ],
            },
            sequence_id,
            attempt_index,
            prefix="v2-sequence",
            output_dir=resolved_output_dir,
        )
        return result.overall_score

    @kbench.task(name="adaptive_shift_v2_overall")
    def adaptive_shift_v2_overall(llm) -> float:
        results = []
        for sequence in build_v2_sequences():
            for attempt_index in range(DEFAULT_ATTEMPTS):
                with kbench.chats.new(f"{sequence.id}-attempt-{attempt_index + 1}", orphan=True):
                    results.append(
                        run_sequence(
                            KBenchAdapter(llm),
                            sequence,
                            attempt_index=attempt_index,
                            prompt_style="release_note",
                        )
                    )
        report = aggregate_sequence_results(results)
        write_sequence_report_bundle(report, resolved_output_dir)
        return float(report["metrics"]["overall"])

    return adaptive_shift_v2_attempt, adaptive_shift_v2_sequence, adaptive_shift_v2_overall


def build_kbench_v2_learning_tasks(output_dir: str | Path | None = None):
    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kaggle_benchmarks is required to build Kaggle tasks. "
            "Import this function inside a Kaggle notebook or an environment with kaggle_benchmarks installed."
        ) from exc
    resolved_output_dir = _resolve_output_dir(output_dir) / "v2_learning"

    @kbench.task(name="adaptive_shift_v2_learning_attempt")
    def adaptive_shift_v2_learning_attempt(llm, scenario_id: str, attempt_index: int = 0) -> float:
        scenario = get_v2_learning_scenario(scenario_id)
        result = run_scenario(
            KBenchAdapter(llm),
            scenario,
            attempt_index=attempt_index,
            prompt_style="release_note",
        )
        _write_attempt_report(
            {
                "scenario_id": result.scenario_id,
                "sequence_id": result.sequence_id,
                "sequence_stage": result.sequence_stage.value if result.sequence_stage is not None else None,
                "family": result.family.value,
                "attempt_index": result.attempt_index,
                "passed": result.passed,
                "score": result.score,
                "failure_tags": result.failure_tags,
                "selected_entities": result.selected_entities,
                "used_evidence_ids": result.used_evidence_ids,
                "surfaced_evidence_ids": result.surfaced_evidence_ids,
                "score_breakdown": result.score_breakdown,
                "trace_digest": result.trace_digest,
            },
            scenario_id,
            attempt_index,
            prefix="v2-learning-scenario",
            output_dir=resolved_output_dir,
        )
        return result.score

    @kbench.task(name="adaptive_shift_v2_learning_sequence")
    def adaptive_shift_v2_learning_sequence(llm, sequence_id: str, attempt_index: int = 0) -> float:
        sequence = get_v2_learning_sequence(sequence_id)
        with kbench.chats.new(f"{sequence.id}-attempt-{attempt_index + 1}", orphan=True):
            result = run_sequence(
                KBenchAdapter(llm),
                sequence,
                attempt_index=attempt_index,
                prompt_style="release_note",
            )
        _write_attempt_report(
            {
                "sequence_id": result.sequence_id,
                "family": result.family.value,
                "attempt_index": result.attempt_index,
                "passed": result.passed,
                "overall_score": result.overall_score,
                "learning_score": result.learning_score,
                "score_breakdown": result.score_breakdown,
                "learned_rules": result.learned_rules,
                "stage_results": [
                    {
                        "scenario_id": stage.scenario_id,
                        "sequence_stage": stage.sequence_stage.value if stage.sequence_stage is not None else None,
                        "score": stage.score,
                        "passed": stage.passed,
                        "score_breakdown": stage.score_breakdown,
                        "used_evidence_ids": stage.used_evidence_ids,
                        "surfaced_evidence_ids": stage.surfaced_evidence_ids,
                    }
                    for stage in result.stage_results
                ],
            },
            sequence_id,
            attempt_index,
            prefix="v2-learning-sequence",
            output_dir=resolved_output_dir,
        )
        return result.learning_score

    @kbench.task(name="adaptive_shift_v2_learning_overall")
    def adaptive_shift_v2_learning_overall(llm) -> float:
        results = []
        for sequence in build_v2_learning_sequences():
            for attempt_index in range(DEFAULT_ATTEMPTS):
                with kbench.chats.new(f"{sequence.id}-attempt-{attempt_index + 1}", orphan=True):
                    results.append(
                        run_sequence(
                            KBenchAdapter(llm),
                            sequence,
                            attempt_index=attempt_index,
                            prompt_style="release_note",
                        )
                    )
        report = aggregate_sequence_results(results)
        write_sequence_report_bundle(report, resolved_output_dir)
        return float(report["metrics"]["learning_score"])

    return (
        adaptive_shift_v2_learning_attempt,
        adaptive_shift_v2_learning_sequence,
        adaptive_shift_v2_learning_overall,
    )
