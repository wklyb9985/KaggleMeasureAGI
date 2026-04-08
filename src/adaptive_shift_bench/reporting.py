from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from adaptive_shift_bench.models import EpisodeResult, FEEDBACK_ONLY_POLICY, SequenceResult


V2_OVERALL_WEIGHTS = {
    "semantic_correctness": 0.35,
    "in_task_adaptation": 0.25,
    "cross_task_transfer": 0.25,
    "protocol_compliance": 0.10,
    "efficiency": 0.05,
}
V2_BENCHMARK_VERSION = "2.1"
V2_COMPARABILITY = "not-comparable-to-pre-fix-v2"
V2_LEARNING_BENCHMARK_VERSION = "2.4-learning-b-expanded"
V2_LEARNING_COMPARABILITY = "not-comparable-to-prior-v2.3-learning-b-or-earlier-learning-variants"
V2_LEARNING_SCORE_WEIGHTS = {
    "revision_success": 0.55,
    "transfer_after_revision": 0.30,
    "localized_generalization": 0.15,
}


def _group_by_scenario(results: Iterable[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.scenario_id].append(result)
    for scenario_results in grouped.values():
        scenario_results.sort(key=lambda item: item.attempt_index)
    return dict(grouped)


def _group_by_sequence(results: Iterable[SequenceResult]) -> dict[str, list[SequenceResult]]:
    grouped: dict[str, list[SequenceResult]] = defaultdict(list)
    for result in results:
        grouped[result.sequence_id].append(result)
    for sequence_results in grouped.values():
        sequence_results.sort(key=lambda item: item.attempt_index)
    return dict(grouped)


def _compute_core_metrics(grouped: dict[str, list[EpisodeResult]]) -> dict[str, float]:
    if not grouped:
        return {"pass_at_1": 0.0, "pass_at_5": 0.0, "avg5": 0.0, "overall": 0.0}

    pass_at_1 = sum(1.0 for runs in grouped.values() if runs and runs[0].passed) / len(grouped)
    pass_at_5 = sum(1.0 for runs in grouped.values() if any(run.passed for run in runs[:5])) / len(grouped)
    avg5 = sum(sum(run.score for run in runs[:5]) / max(1, len(runs[:5])) for runs in grouped.values()) / len(grouped)
    overall = 0.50 * pass_at_1 + 0.30 * pass_at_5 + 0.20 * avg5
    return {
        "pass_at_1": pass_at_1,
        "pass_at_5": pass_at_5,
        "avg5": avg5,
        "overall": overall,
    }


def _compute_sequence_metrics(grouped: dict[str, list[SequenceResult]]) -> dict[str, float]:
    if not grouped:
        return {
            "pass_at_1": 0.0,
            "pass_at_5": 0.0,
            "avg5": 0.0,
            "overall": 0.0,
            "semantic_correctness": 0.0,
            "in_task_adaptation": 0.0,
            "cross_task_transfer": 0.0,
            "protocol_compliance": 0.0,
            "efficiency": 0.0,
        }

    first_runs = [runs[0] for runs in grouped.values() if runs]
    benchmark_suite = first_runs[0].benchmark_suite if first_runs else "v2"
    pass_at_1 = sum(1.0 for result in first_runs if result.passed) / len(grouped)
    pass_at_5 = sum(1.0 for runs in grouped.values() if any(run.passed for run in runs[:5])) / len(grouped)
    if benchmark_suite == "v2_learning":
        avg5 = sum(sum(run.learning_score for run in runs[:5]) / max(1, len(runs[:5])) for runs in grouped.values()) / len(grouped)
    else:
        avg5 = sum(sum(run.overall_score for run in runs[:5]) / max(1, len(runs[:5])) for runs in grouped.values()) / len(grouped)
    semantic_correctness = sum(result.semantic_correctness for result in first_runs) / len(first_runs)
    in_task_adaptation = sum(result.in_task_adaptation for result in first_runs) / len(first_runs)
    cross_task_transfer = sum(result.cross_task_transfer for result in first_runs) / len(first_runs)
    protocol_compliance = sum(result.protocol_compliance for result in first_runs) / len(first_runs)
    efficiency = sum(result.efficiency_score for result in first_runs) / len(first_runs)
    metrics = {
        "pass_at_1": pass_at_1,
        "pass_at_5": pass_at_5,
        "avg5": avg5,
        "overall": 0.0,
        "semantic_correctness": semantic_correctness,
        "in_task_adaptation": in_task_adaptation,
        "cross_task_transfer": cross_task_transfer,
        "protocol_compliance": protocol_compliance,
        "efficiency": efficiency,
    }
    if benchmark_suite == "v2_learning":
        learning_score = sum(result.learning_score for result in first_runs) / len(first_runs)
        metrics.update(
            {
                "prior_probe_correctness": sum(result.prior_probe_correctness for result in first_runs) / len(first_runs),
                "stale_prior_rate": sum(result.stale_prior_rate for result in first_runs) / len(first_runs),
                "revision_success": sum(result.revision_success for result in first_runs) / len(first_runs),
                "revision_turns_to_fix": sum(result.revision_turns_to_fix for result in first_runs) / len(first_runs),
                "revision_efficiency": sum(result.revision_efficiency for result in first_runs) / len(first_runs),
                "transfer_after_revision": sum(result.transfer_after_revision for result in first_runs) / len(first_runs),
                "localized_generalization": sum(result.localized_generalization for result in first_runs) / len(first_runs),
                "learning_score": learning_score,
            }
        )
        metrics["overall"] = learning_score
    else:
        metrics["overall"] = (
            V2_OVERALL_WEIGHTS["semantic_correctness"] * semantic_correctness
            + V2_OVERALL_WEIGHTS["in_task_adaptation"] * in_task_adaptation
            + V2_OVERALL_WEIGHTS["cross_task_transfer"] * cross_task_transfer
            + V2_OVERALL_WEIGHTS["protocol_compliance"] * protocol_compliance
            + V2_OVERALL_WEIGHTS["efficiency"] * efficiency
        )
    return metrics


def _bootstrap_ci(
    grouped: dict[str, list[EpisodeResult]],
    metric_name: str,
    *,
    samples: int = 400,
    seed: int = 17,
) -> tuple[float, float]:
    if not grouped:
        return 0.0, 0.0

    scenario_ids = list(grouped)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        resampled_ids = [rng.choice(scenario_ids) for _ in scenario_ids]
        resampled_grouped = {f"{idx}-{scenario_id}": grouped[scenario_id] for idx, scenario_id in enumerate(resampled_ids)}
        values.append(_compute_core_metrics(resampled_grouped)[metric_name])
    values.sort()
    lower_index = int(0.025 * (len(values) - 1))
    upper_index = int(0.975 * (len(values) - 1))
    return values[lower_index], values[upper_index]


def _bootstrap_sequence_ci(
    grouped: dict[str, list[SequenceResult]],
    metric_name: str,
    *,
    samples: int = 400,
    seed: int = 17,
) -> tuple[float, float]:
    if not grouped:
        return 0.0, 0.0

    sequence_ids = list(grouped)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        resampled_ids = [rng.choice(sequence_ids) for _ in sequence_ids]
        resampled_grouped = {f"{idx}-{sequence_id}": grouped[sequence_id] for idx, sequence_id in enumerate(resampled_ids)}
        values.append(_compute_sequence_metrics(resampled_grouped)[metric_name])
    values.sort()
    lower_index = int(0.025 * (len(values) - 1))
    upper_index = int(0.975 * (len(values) - 1))
    return values[lower_index], values[upper_index]


def aggregate_attempts(results: Iterable[EpisodeResult]) -> dict[str, object]:
    grouped = _group_by_scenario(results)
    metrics = _compute_core_metrics(grouped)

    scenario_breakdown: dict[str, dict[str, object]] = {}
    family_grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for scenario_id, runs in grouped.items():
        family = runs[0].family.value if runs else "unknown"
        family_grouped[family].extend(runs)
        scenario_breakdown[scenario_id] = {
            "family": family,
            "pass_at_1": bool(runs and runs[0].passed),
            "pass_at_5": any(run.passed for run in runs[:5]),
            "avg5": sum(run.score for run in runs[:5]) / max(1, len(runs[:5])),
            "attempts": [asdict(run) for run in runs],
        }

    family_breakdown: dict[str, dict[str, float]] = {}
    for family, family_results in family_grouped.items():
        family_breakdown[family] = _compute_core_metrics(_group_by_scenario(family_results))

    confidence_intervals = {
        key: _bootstrap_ci(grouped, key)
        for key in ("pass_at_1", "pass_at_5", "avg5", "overall")
    }

    return {
        "metrics": metrics,
        "benchmark_metadata": {
            "feedback_only_policy": FEEDBACK_ONLY_POLICY,
            "report_mode": "v1_attempts",
        },
        "confidence_intervals": confidence_intervals,
        "family_breakdown": family_breakdown,
        "scenario_breakdown": scenario_breakdown,
    }


def aggregate_sequence_results(results: Iterable[SequenceResult]) -> dict[str, object]:
    grouped = _group_by_sequence(results)
    metrics = _compute_sequence_metrics(grouped)
    first_result = next(iter(next(iter(grouped.values()))), None) if grouped else None
    benchmark_suite = first_result.benchmark_suite if first_result is not None else "v2"

    sequence_breakdown: dict[str, dict[str, object]] = {}
    family_grouped: dict[str, list[SequenceResult]] = defaultdict(list)
    for sequence_id, runs in grouped.items():
        family = runs[0].family.value if runs else "unknown"
        family_grouped[family].extend(runs)
        sequence_breakdown[sequence_id] = {
            "family": family,
            "pass_at_1": bool(runs and runs[0].passed),
            "pass_at_5": any(run.passed for run in runs[:5]),
            "avg5": (
                sum(
                    (
                        run.learning_score
                        if (runs and runs[0].benchmark_suite == "v2_learning")
                        else run.overall_score
                    )
                    for run in runs[:5]
                )
                / max(1, len(runs[:5]))
            ),
            "benchmark_suite": runs[0].benchmark_suite if runs else benchmark_suite,
            "attempts": [asdict(run) for run in runs],
        }

    family_breakdown: dict[str, dict[str, float]] = {}
    for family, family_results in family_grouped.items():
        family_breakdown[family] = _compute_sequence_metrics(_group_by_sequence(family_results))

    ci_keys = [
        "pass_at_1",
        "pass_at_5",
        "avg5",
        "overall",
        "semantic_correctness",
        "in_task_adaptation",
        "cross_task_transfer",
        "protocol_compliance",
        "efficiency",
    ]
    if benchmark_suite == "v2_learning":
        ci_keys.extend(
            [
                "prior_probe_correctness",
                "stale_prior_rate",
                "revision_success",
                "revision_turns_to_fix",
                "revision_efficiency",
                "transfer_after_revision",
                "localized_generalization",
                "learning_score",
            ]
        )
    confidence_intervals = {key: _bootstrap_sequence_ci(grouped, key) for key in ci_keys}

    return {
        "metrics": metrics,
        "benchmark_metadata": {
            "benchmark_suite": benchmark_suite,
            "benchmark_version": (
                V2_LEARNING_BENCHMARK_VERSION
                if benchmark_suite == "v2_learning"
                else V2_BENCHMARK_VERSION
            ),
            "comparability": (
                V2_LEARNING_COMPARABILITY
                if benchmark_suite == "v2_learning"
                else V2_COMPARABILITY
            ),
            "feedback_only_policy": FEEDBACK_ONLY_POLICY,
            "report_mode": "v2_learning_sequences" if benchmark_suite == "v2_learning" else "v2_sequences",
            "learning_variant": "b_expanded" if benchmark_suite == "v2_learning" else None,
            "overall_weights": (
                dict(V2_LEARNING_SCORE_WEIGHTS) if benchmark_suite == "v2_learning" else dict(V2_OVERALL_WEIGHTS)
            ),
            "primary_leaderboard_metric": "learning_score" if benchmark_suite == "v2_learning" else "overall",
            "learning_score_weights": dict(V2_LEARNING_SCORE_WEIGHTS) if benchmark_suite == "v2_learning" else {},
        },
        "confidence_intervals": confidence_intervals,
        "family_breakdown": family_breakdown,
        "sequence_breakdown": sequence_breakdown,
    }


def write_report_bundle(report: dict[str, object], output_dir: str | Path = "reports") -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "adaptive_shift_report.json"
    md_path = output_path / "adaptive_shift_report.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    metrics = report["metrics"]
    ci = report["confidence_intervals"]
    markdown = "\n".join(
        [
            "# Adaptive Shift Report",
            "",
            f"- feedback_only_policy: {report['benchmark_metadata']['feedback_only_policy']}",
            f"- pass@1: {metrics['pass_at_1']:.3f} ({ci['pass_at_1'][0]:.3f}, {ci['pass_at_1'][1]:.3f})",
            f"- pass@5: {metrics['pass_at_5']:.3f} ({ci['pass_at_5'][0]:.3f}, {ci['pass_at_5'][1]:.3f})",
            f"- avg5: {metrics['avg5']:.3f} ({ci['avg5'][0]:.3f}, {ci['avg5'][1]:.3f})",
            f"- overall: {metrics['overall']:.3f} ({ci['overall'][0]:.3f}, {ci['overall'][1]:.3f})",
            "",
            "## Families",
            "",
            *[
                f"- {family}: pass@1={values['pass_at_1']:.3f}, pass@5={values['pass_at_5']:.3f}, avg5={values['avg5']:.3f}, overall={values['overall']:.3f}"
                for family, values in sorted(report["family_breakdown"].items())
            ],
        ]
    )
    md_path.write_text(markdown + "\n", encoding="utf-8")
    return json_path, md_path


def write_sequence_report_bundle(report: dict[str, object], output_dir: str | Path = "reports") -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark_metadata = report.get("benchmark_metadata", {})
    report_mode = benchmark_metadata.get("report_mode", "v2_sequences")
    if report_mode == "v2_learning_sequences":
        json_path = output_path / "adaptive_shift_v2_learning_report.json"
        md_path = output_path / "adaptive_shift_v2_learning_report.md"
        title = "# Adaptive Shift V2 Learning Report"
    else:
        json_path = output_path / "adaptive_shift_v2_report.json"
        md_path = output_path / "adaptive_shift_v2_report.md"
        title = "# Adaptive Shift V2 Report"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    metrics = report["metrics"]
    ci = report["confidence_intervals"]
    benchmark_version = benchmark_metadata.get("benchmark_version", "obsolete_pre_fix_v2")
    comparability = benchmark_metadata.get("comparability", "not-comparable-to-v2.1")
    lines = [
        title,
        "",
        f"- benchmark_version: {benchmark_version}",
        f"- comparability: {comparability}",
        f"- benchmark_suite: {benchmark_metadata.get('benchmark_suite', 'v2')}",
        f"- primary_leaderboard_metric: {benchmark_metadata.get('primary_leaderboard_metric', 'overall')}",
        f"- feedback_only_policy: {report['benchmark_metadata']['feedback_only_policy']}",
        f"- semantic_correctness: {metrics['semantic_correctness']:.3f} ({ci['semantic_correctness'][0]:.3f}, {ci['semantic_correctness'][1]:.3f})",
        f"- in_task_adaptation: {metrics['in_task_adaptation']:.3f} ({ci['in_task_adaptation'][0]:.3f}, {ci['in_task_adaptation'][1]:.3f})",
        f"- cross_task_transfer: {metrics['cross_task_transfer']:.3f} ({ci['cross_task_transfer'][0]:.3f}, {ci['cross_task_transfer'][1]:.3f})",
        f"- protocol_compliance: {metrics['protocol_compliance']:.3f} ({ci['protocol_compliance'][0]:.3f}, {ci['protocol_compliance'][1]:.3f})",
        f"- efficiency: {metrics['efficiency']:.3f} ({ci['efficiency'][0]:.3f}, {ci['efficiency'][1]:.3f})",
        f"- overall: {metrics['overall']:.3f} ({ci['overall'][0]:.3f}, {ci['overall'][1]:.3f})",
        f"- pass@1: {metrics['pass_at_1']:.3f} ({ci['pass_at_1'][0]:.3f}, {ci['pass_at_1'][1]:.3f})",
        f"- pass@5: {metrics['pass_at_5']:.3f} ({ci['pass_at_5'][0]:.3f}, {ci['pass_at_5'][1]:.3f})",
        f"- avg5: {metrics['avg5']:.3f} ({ci['avg5'][0]:.3f}, {ci['avg5'][1]:.3f})",
    ]
    if report_mode == "v2_learning_sequences":
        lines.extend(
            [
                f"- prior_probe_correctness: {metrics['prior_probe_correctness']:.3f} ({ci['prior_probe_correctness'][0]:.3f}, {ci['prior_probe_correctness'][1]:.3f})",
                f"- stale_prior_rate: {metrics['stale_prior_rate']:.3f} ({ci['stale_prior_rate'][0]:.3f}, {ci['stale_prior_rate'][1]:.3f})",
                f"- revision_success: {metrics['revision_success']:.3f} ({ci['revision_success'][0]:.3f}, {ci['revision_success'][1]:.3f})",
                f"- revision_turns_to_fix: {metrics['revision_turns_to_fix']:.3f} ({ci['revision_turns_to_fix'][0]:.3f}, {ci['revision_turns_to_fix'][1]:.3f})",
                f"- revision_efficiency: {metrics['revision_efficiency']:.3f} ({ci['revision_efficiency'][0]:.3f}, {ci['revision_efficiency'][1]:.3f})",
                f"- transfer_after_revision: {metrics['transfer_after_revision']:.3f} ({ci['transfer_after_revision'][0]:.3f}, {ci['transfer_after_revision'][1]:.3f})",
                f"- localized_generalization: {metrics['localized_generalization']:.3f} ({ci['localized_generalization'][0]:.3f}, {ci['localized_generalization'][1]:.3f})",
                f"- learning_score: {metrics['learning_score']:.3f} ({ci['learning_score'][0]:.3f}, {ci['learning_score'][1]:.3f})",
            ]
        )
    lines.extend(
        [
            "",
            "## Families",
            "",
            *[
                (
                    f"- {family}: learning_score={values['learning_score']:.3f}, revision={values['revision_success']:.3f}, "
                    f"transfer_after_revision={values['transfer_after_revision']:.3f}, overall={values['overall']:.3f}"
                    if report_mode == "v2_learning_sequences"
                    else f"- {family}: semantic={values['semantic_correctness']:.3f}, transfer={values['cross_task_transfer']:.3f}, protocol={values['protocol_compliance']:.3f}, overall={values['overall']:.3f}"
                )
                for family, values in sorted(report["family_breakdown"].items())
            ],
        ]
    )
    markdown = "\n".join(lines)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    return json_path, md_path
