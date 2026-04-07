from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from adaptive_shift_bench.models import EpisodeResult


def _group_by_scenario(results: Iterable[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.scenario_id].append(result)
    for scenario_results in grouped.values():
        scenario_results.sort(key=lambda item: item.attempt_index)
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
        "confidence_intervals": confidence_intervals,
        "family_breakdown": family_breakdown,
        "scenario_breakdown": scenario_breakdown,
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

