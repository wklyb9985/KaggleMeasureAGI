from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_shift_bench.claude_cli import ClaudeCLIAdapter
from adaptive_shift_bench.engine import PromptStyle, run_scenario
from adaptive_shift_bench.reporting import aggregate_attempts, write_report_bundle
from adaptive_shift_bench.scenarios import get_scenario

PILOT_SCENARIO_IDS = (
    "api_migration-easy-explicit_change",
    "api_migration-hard-docs_search",
    "dsl_wrapper-easy-docs_inline",
    "dsl_wrapper-hard-feedback_only",
    "future_registry-easy-explicit_change",
    "future_registry-hard-feedback_only",
)


def build_pilot_suite():
    return tuple(get_scenario(scenario_id) for scenario_id in PILOT_SCENARIO_IDS)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _attempt_payload(result, adapter: ClaudeCLIAdapter) -> dict:
    return {
        "scenario_id": result.scenario_id,
        "family": result.family.value,
        "attempt_index": result.attempt_index,
        "passed": result.passed,
        "score": result.score,
        "failure_tags": list(result.failure_tags),
        "selected_entities": list(result.selected_entities),
        "used_evidence_ids": list(result.used_evidence_ids),
        "trace_digest": result.trace_digest,
        "functional_correctness": result.functional_correctness,
        "adaptation_correctness": result.adaptation_correctness,
        "legacy_avoidance": result.legacy_avoidance,
        "claude": {
            "model_alias": adapter.model,
            "turn_count": len(adapter.invocations),
            "total_cost_usd": sum(item.total_cost_usd for item in adapter.invocations),
            "invocations": [
                {
                    "duration_ms": item.duration_ms,
                    "total_cost_usd": item.total_cost_usd,
                    "session_id": item.session_id,
                    "model_usage": item.model_usage,
                    "response_text": item.response_text,
                }
                for item in adapter.invocations
            ],
        },
    }


def run_claude_pilot(
    *,
    model: str,
    attempts: int,
    workspace_dir: str | Path,
    max_budget_usd: float,
    effort: str,
    prompt_style: PromptStyle,
) -> dict[str, object]:
    workspace_path = Path(workspace_dir)
    model_dir = workspace_path / prompt_style / model
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    attempt_reports = []
    for scenario in build_pilot_suite():
        for attempt_index in range(attempts):
            adapter = ClaudeCLIAdapter(
                model=model,
                workspace_dir=model_dir,
                max_budget_usd=max_budget_usd,
                effort=effort,
            )
            result = run_scenario(
                adapter,
                scenario,
                attempt_index=attempt_index,
                prompt_style=prompt_style,
            )
            results.append(result)
            payload = _attempt_payload(result, adapter)
            attempt_reports.append(payload)
            attempt_path = model_dir / "attempts" / f"{scenario.id}-attempt-{attempt_index + 1}.json"
            _write_json(attempt_path, payload)

    report = aggregate_attempts(results)
    report["runner"] = {
        "mode": "claude-cli-pilot",
        "model_alias": model,
        "prompt_style": prompt_style,
        "attempts_per_scenario": attempts,
        "scenario_ids": list(PILOT_SCENARIO_IDS),
        "total_cli_cost_usd": sum(item["claude"]["total_cost_usd"] for item in attempt_reports),
        "total_turns": sum(item["claude"]["turn_count"] for item in attempt_reports),
    }
    write_report_bundle(report, model_dir / "report")
    _write_json(model_dir / "report" / "claude_pilot_full.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Claude CLI pilot against the adaptive shift benchmark.")
    parser.add_argument("--models", nargs="+", default=["sonnet", "haiku"])
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--workspace-dir", default="workspace/claude_cli_pilot")
    parser.add_argument("--max-budget-usd", type=float, default=1.50)
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "max"))
    parser.add_argument("--prompt-style", default="release_note", choices=("benchmark", "release_note"))
    args = parser.parse_args()

    summary = {}
    for model in args.models:
        summary[model] = run_claude_pilot(
            model=model,
            attempts=args.attempts,
            workspace_dir=args.workspace_dir,
            max_budget_usd=args.max_budget_usd,
            effort=args.effort,
            prompt_style=args.prompt_style,
        )["metrics"]

    print(json.dumps(summary, indent=2, sort_keys=True))
