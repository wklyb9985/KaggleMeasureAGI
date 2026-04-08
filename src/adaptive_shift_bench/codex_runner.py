from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from adaptive_shift_bench.claude_runner import PILOT_SCENARIO_IDS, build_pilot_suite
from adaptive_shift_bench.codex_cli import CodexCLIAdapter
from adaptive_shift_bench.engine import PromptStyle, run_scenario
from adaptive_shift_bench.models import FEEDBACK_ONLY_POLICY
from adaptive_shift_bench.reporting import aggregate_attempts, write_report_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _attempt_payload(result, adapter: CodexCLIAdapter) -> dict:
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
        "codex": {
            "model_alias": adapter.model,
            "reasoning_effort": adapter.effort,
            "turn_count": len(adapter.invocations),
            "total_input_tokens": sum(item.input_tokens for item in adapter.invocations),
            "total_cached_input_tokens": sum(item.cached_input_tokens for item in adapter.invocations),
            "total_output_tokens": sum(item.output_tokens for item in adapter.invocations),
            "invocations": [
                {
                    "duration_ms": item.duration_ms,
                    "input_tokens": item.input_tokens,
                    "cached_input_tokens": item.cached_input_tokens,
                    "output_tokens": item.output_tokens,
                    "thread_id": item.thread_id,
                    "response_text": item.response_text,
                }
                for item in adapter.invocations
            ],
        },
    }


def run_codex_pilot(
    *,
    model: str,
    attempts: int,
    workspace_dir: str | Path,
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
            with tempfile.TemporaryDirectory(prefix="adaptive-shift-codex-") as temp_dir:
                adapter = CodexCLIAdapter(
                    model=model,
                    workspace_dir=temp_dir,
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
        "mode": "codex-cli-pilot",
        "model_alias": model,
        "prompt_style": prompt_style,
        "attempts_per_scenario": attempts,
        "scenario_ids": list(PILOT_SCENARIO_IDS),
        "feedback_only_policy": FEEDBACK_ONLY_POLICY,
        "total_turns": sum(item["codex"]["turn_count"] for item in attempt_reports),
        "total_input_tokens": sum(item["codex"]["total_input_tokens"] for item in attempt_reports),
        "total_cached_input_tokens": sum(item["codex"]["total_cached_input_tokens"] for item in attempt_reports),
        "total_output_tokens": sum(item["codex"]["total_output_tokens"] for item in attempt_reports),
    }
    write_report_bundle(report, model_dir / "report")
    _write_json(model_dir / "report" / "codex_pilot_full.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Codex CLI pilot against the adaptive shift benchmark.")
    parser.add_argument("--models", nargs="+", default=["gpt-5.4", "gpt-5.4-mini"])
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--workspace-dir", default="workspace/codex_cli_pilot")
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--prompt-style", default="release_note", choices=("benchmark", "release_note"))
    args = parser.parse_args()

    summary = {}
    for model in args.models:
        summary[model] = run_codex_pilot(
            model=model,
            attempts=args.attempts,
            workspace_dir=args.workspace_dir,
            effort=args.effort,
            prompt_style=args.prompt_style,
        )["metrics"]

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
