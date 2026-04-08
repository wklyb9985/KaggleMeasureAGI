"""Run v2 sequences against a single model and save results."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adaptive_shift_bench.engine import run_sequence
from adaptive_shift_bench.reporting import aggregate_sequence_results, write_sequence_report_bundle
from adaptive_shift_bench.scenarios import build_v2_learning_sequences, build_v2_sequences


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_v2(
    *,
    backend: str,
    model: str,
    effort: str,
    workspace_dir: str,
    suite: str = "v2",
    attempts: int = 1,
    max_budget_usd: float = 3.00,
) -> dict:
    workspace_path = Path(workspace_dir) / f"{model}-{effort}"
    workspace_path.mkdir(parents=True, exist_ok=True)

    sequences = build_v2_learning_sequences() if suite == "v2_learning" else build_v2_sequences()
    all_results = []

    for sequence in sequences:
        for attempt_index in range(attempts):
            with tempfile.TemporaryDirectory(prefix=f"asb-v2-{backend}-") as temp_dir:
                if backend == "claude":
                    from adaptive_shift_bench.claude_cli import ClaudeCLIAdapter
                    adapter = ClaudeCLIAdapter(
                        model=model,
                        workspace_dir=temp_dir,
                        max_budget_usd=max_budget_usd,
                        effort=effort,
                    )
                else:
                    from adaptive_shift_bench.codex_cli import CodexCLIAdapter
                    adapter = CodexCLIAdapter(
                        model=model,
                        workspace_dir=temp_dir,
                        effort=effort,
                    )

                result = run_sequence(adapter, sequence, attempt_index=attempt_index)

            all_results.append(result)

            # Save per-sequence result
            seq_data = asdict(result)
            seq_path = workspace_path / "sequences" / f"{sequence.id}-attempt-{attempt_index + 1}.json"
            _write_json(seq_path, seq_data)

            # Print progress
            stage_pass = [r.passed for r in result.stage_results]
            print(
                f"  {sequence.id}: passed={result.passed} overall={result.overall_score:.3f} "
                f"semantic={result.semantic_correctness:.2f} transfer={result.cross_task_transfer:.2f} "
                f"stages={''.join('P' if p else 'F' for p in stage_pass)} "
                f"learned={list(result.learned_rules)}",
                flush=True,
            )

    report = aggregate_sequence_results(all_results)
    write_sequence_report_bundle(report, workspace_path / "report")
    report_name = "v2_learning_full.json" if suite == "v2_learning" else "v2_full.json"
    _write_json(workspace_path / "report" / report_name, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("claude", "codex"))
    parser.add_argument("--suite", default="v2", choices=("v2", "v2_learning"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-budget-usd", type=float, default=3.00)
    args = parser.parse_args()

    workspace_dir = args.workspace_dir or (
        "workspace/v2_learning_bench_v2_4_b_expanded" if args.suite == "v2_learning" else "workspace/v2_bench_v2_1"
    )
    label = "V2 Learning Sequences" if args.suite == "v2_learning" else "V2 Sequences"
    print(f"\n=== {label}: {args.model} ({args.effort}) ===", flush=True)
    report = run_v2(
        backend=args.backend,
        model=args.model,
        effort=args.effort,
        workspace_dir=workspace_dir,
        suite=args.suite,
        attempts=args.attempts,
        max_budget_usd=args.max_budget_usd,
    )
    print(f"\n--- Metrics for {args.model} ({args.effort}) [{args.suite}] ---")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
