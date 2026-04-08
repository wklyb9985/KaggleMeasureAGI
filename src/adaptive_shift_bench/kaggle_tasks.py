import json
import os
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adaptive_shift_bench.engine import run_scenario, run_sequence
from adaptive_shift_bench.models import (
    ActionRecord,
    EpisodeAction,
    EpisodeResult,
    ScenarioFamily,
    SequenceResult,
    SequenceStage,
)
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
_PUBLIC_V2_LEARNING_SEQUENCE_IDS = {
    "adaptive_shift_v2_learning_openai": "v2-learning-openai-revision",
    "adaptive_shift_v2_learning_pandas": "v2-learning-pandas-revision",
    "adaptive_shift_v2_learning_registry": "v2-learning-registry-revision",
}
adaptive_shift_v2_learning_openai = None
adaptive_shift_v2_learning_pandas = None
adaptive_shift_v2_learning_registry = None
adaptive_shift_v2_learning_overall = None


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
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _public_sequence_report_path(
    sequence_id: str,
    attempt_index: int,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    base = _resolve_output_dir(output_dir) / "v2_learning"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"public-sequence-{sequence_id}-attempt-{attempt_index + 1}.json"


def _write_public_sequence_report(
    result: SequenceResult,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    path = _public_sequence_report_path(result.sequence_id, result.attempt_index, output_dir=output_dir)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _episode_result_from_dict(data: dict[str, Any]) -> EpisodeResult:
    action_records = tuple(
        ActionRecord(
            turn_index=record["turn_index"],
            action=EpisodeAction(
                action=record["action"]["action"],
                query=record["action"].get("query"),
                doc_id=record["action"].get("doc_id"),
                candidate=record["action"].get("candidate"),
                model=record["action"].get("model"),
                content=record["action"].get("content"),
                raw_response=record["action"].get("raw_response", ""),
            ),
            observation=record["observation"],
        )
        for record in data.get("action_records", ())
    )
    sequence_stage = data.get("sequence_stage")
    return EpisodeResult(
        scenario_id=data["scenario_id"],
        family=ScenarioFamily(data["family"]),
        attempt_index=data["attempt_index"],
        passed=data["passed"],
        score=data["score"],
        failure_tags=tuple(data.get("failure_tags", ())),
        selected_entities=tuple(data.get("selected_entities", ())),
        used_evidence_ids=tuple(data.get("used_evidence_ids", ())),
        trace_digest=data.get("trace_digest", ""),
        action_records=action_records,
        functional_correctness=data.get("functional_correctness", False),
        adaptation_correctness=data.get("adaptation_correctness", False),
        legacy_avoidance=data.get("legacy_avoidance", False),
        semantic_correctness=data.get("semantic_correctness", 0.0),
        protocol_compliance=data.get("protocol_compliance", 1.0),
        efficiency_score=data.get("efficiency_score", 1.0),
        learning_transfer_score=data.get("learning_transfer_score", 0.0),
        turn_count=data.get("turn_count", 0),
        evidence_action_count=data.get("evidence_action_count", 0),
        surfaced_evidence_ids=tuple(data.get("surfaced_evidence_ids", ())),
        benchmark_suite=data.get("benchmark_suite", "v1"),
        sequence_id=data.get("sequence_id"),
        sequence_stage=SequenceStage(sequence_stage) if sequence_stage else None,
        score_breakdown=dict(data.get("score_breakdown", {})),
    )


def _sequence_result_from_dict(data: dict[str, Any]) -> SequenceResult:
    return SequenceResult(
        sequence_id=data["sequence_id"],
        family=ScenarioFamily(data["family"]),
        attempt_index=data["attempt_index"],
        passed=data["passed"],
        overall_score=data["overall_score"],
        stage_results=tuple(
            _episode_result_from_dict(stage_data) for stage_data in data.get("stage_results", ())
        ),
        in_task_adaptation=data.get("in_task_adaptation", 0.0),
        cross_task_transfer=data.get("cross_task_transfer", 0.0),
        semantic_correctness=data.get("semantic_correctness", 0.0),
        protocol_compliance=data.get("protocol_compliance", 0.0),
        efficiency_score=data.get("efficiency_score", 0.0),
        benchmark_suite=data.get("benchmark_suite", "v2"),
        learned_rules=tuple(data.get("learned_rules", ())),
        prior_probe_correctness=data.get("prior_probe_correctness", 0.0),
        stale_prior_rate=data.get("stale_prior_rate", 0.0),
        revision_success=data.get("revision_success", 0.0),
        revision_turns_to_fix=data.get("revision_turns_to_fix", 0.0),
        revision_efficiency=data.get("revision_efficiency", 0.0),
        transfer_after_revision=data.get("transfer_after_revision", 0.0),
        localized_generalization=data.get("localized_generalization", 0.0),
        learning_score=data.get("learning_score", 0.0),
        score_breakdown=dict(data.get("score_breakdown", {})),
    )


def _load_public_sequence_result(
    sequence_id: str,
    attempt_index: int,
    *,
    output_dir: str | Path | None = None,
) -> SequenceResult:
    path = _public_sequence_report_path(sequence_id, attempt_index, output_dir=output_dir)
    return _sequence_result_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _run_learning_sequence_result(
    llm: Any,
    sequence_id: str,
    attempt_index: int,
) -> SequenceResult:
    import kaggle_benchmarks as kbench

    sequence = get_v2_learning_sequence(sequence_id)
    with kbench.chats.new(f"{sequence.id}-attempt-{attempt_index + 1}", orphan=True):
        return run_sequence(
            KBenchAdapter(llm),
            sequence,
            attempt_index=attempt_index,
            prompt_style="release_note",
        )


def _adaptive_shift_v2_learning_openai_func(llm, attempt_index: int = 0) -> float:
    result = _run_learning_sequence_result(
        llm,
        _PUBLIC_V2_LEARNING_SEQUENCE_IDS["adaptive_shift_v2_learning_openai"],
        attempt_index,
    )
    _write_public_sequence_report(result)
    return result.learning_score


def _adaptive_shift_v2_learning_pandas_func(llm, attempt_index: int = 0) -> float:
    result = _run_learning_sequence_result(
        llm,
        _PUBLIC_V2_LEARNING_SEQUENCE_IDS["adaptive_shift_v2_learning_pandas"],
        attempt_index,
    )
    _write_public_sequence_report(result)
    return result.learning_score


def _adaptive_shift_v2_learning_registry_func(llm, attempt_index: int = 0) -> float:
    result = _run_learning_sequence_result(
        llm,
        _PUBLIC_V2_LEARNING_SEQUENCE_IDS["adaptive_shift_v2_learning_registry"],
        attempt_index,
    )
    _write_public_sequence_report(result)
    return result.learning_score


def _adaptive_shift_v2_learning_overall_func(llm) -> float:
    results: list[SequenceResult] = []
    for attempt_index in range(DEFAULT_ATTEMPTS):
        adaptive_shift_v2_learning_openai.run(llm=llm, attempt_index=attempt_index)
        results.append(
            _load_public_sequence_result(
                _PUBLIC_V2_LEARNING_SEQUENCE_IDS["adaptive_shift_v2_learning_openai"],
                attempt_index,
            )
        )
        adaptive_shift_v2_learning_pandas.run(llm=llm, attempt_index=attempt_index)
        results.append(
            _load_public_sequence_result(
                _PUBLIC_V2_LEARNING_SEQUENCE_IDS["adaptive_shift_v2_learning_pandas"],
                attempt_index,
            )
        )
        adaptive_shift_v2_learning_registry.run(llm=llm, attempt_index=attempt_index)
        results.append(
            _load_public_sequence_result(
                _PUBLIC_V2_LEARNING_SEQUENCE_IDS["adaptive_shift_v2_learning_registry"],
                attempt_index,
            )
        )
    report = aggregate_sequence_results(results)
    write_sequence_report_bundle(report, _resolve_output_dir() / "v2_learning")
    return float(report["metrics"]["learning_score"])


def get_public_kbench_v2_learning_tasks():
    global adaptive_shift_v2_learning_openai
    global adaptive_shift_v2_learning_pandas
    global adaptive_shift_v2_learning_registry
    global adaptive_shift_v2_learning_overall

    if all(
        task is not None
        for task in (
            adaptive_shift_v2_learning_openai,
            adaptive_shift_v2_learning_pandas,
            adaptive_shift_v2_learning_registry,
            adaptive_shift_v2_learning_overall,
        )
    ):
        return (
            adaptive_shift_v2_learning_openai,
            adaptive_shift_v2_learning_pandas,
            adaptive_shift_v2_learning_registry,
            adaptive_shift_v2_learning_overall,
        )

    try:
        import kaggle_benchmarks as kbench
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "kaggle_benchmarks is required to build Kaggle tasks. "
            "Import this function inside a Kaggle notebook or an environment with kaggle_benchmarks installed."
        ) from exc

    adaptive_shift_v2_learning_openai = kbench.task(name="adaptive_shift_v2_learning_openai")(
        _adaptive_shift_v2_learning_openai_func
    )
    adaptive_shift_v2_learning_pandas = kbench.task(name="adaptive_shift_v2_learning_pandas")(
        _adaptive_shift_v2_learning_pandas_func
    )
    adaptive_shift_v2_learning_registry = kbench.task(name="adaptive_shift_v2_learning_registry")(
        _adaptive_shift_v2_learning_registry_func
    )
    adaptive_shift_v2_learning_overall = kbench.task(name="adaptive_shift_v2_learning_overall")(
        _adaptive_shift_v2_learning_overall_func
    )
    return (
        adaptive_shift_v2_learning_openai,
        adaptive_shift_v2_learning_pandas,
        adaptive_shift_v2_learning_registry,
        adaptive_shift_v2_learning_overall,
    )


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

    @kbench.task(name="adaptive_shift_v2_learning_attempt_local")
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

    @kbench.task(name="adaptive_shift_v2_learning_sequence_local")
    def adaptive_shift_v2_learning_sequence(llm, sequence_id: str, attempt_index: int = 0) -> float:
        result = _run_learning_sequence_result(llm, sequence_id, attempt_index)
        _write_attempt_report(
            asdict(result),
            sequence_id,
            attempt_index,
            prefix="v2-learning-sequence",
            output_dir=resolved_output_dir,
        )
        return result.learning_score

    @kbench.task(name="adaptive_shift_v2_learning_overall_local")
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


try:  # pragma: no cover - exercised in Kaggle and local mock integration tests
    get_public_kbench_v2_learning_tasks()
except ImportError:
    pass
