from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from adaptive_shift_bench.kaggle_tasks import (
    build_kbench_tasks,
    build_kbench_v2_learning_tasks,
    build_kbench_v2_tasks,
)
from adaptive_shift_bench.local_kaggle_mock import LocalTaskLLM, patched_local_kaggle_benchmarks, run_parallel


def _session_workspace(root: Path, session_key: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", session_key).strip("_") or "session"
    path = root / "sessions" / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_llm(backend: str, model: str, effort: str, workspace_root: Path, max_budget_usd: float) -> LocalTaskLLM:
    def factory(session_key: str):
        workspace_dir = _session_workspace(workspace_root, session_key)
        if backend == "claude":
            from adaptive_shift_bench.claude_cli import ClaudeCLIAdapter

            return ClaudeCLIAdapter(
                model=model,
                workspace_dir=workspace_dir,
                max_budget_usd=max_budget_usd,
                effort=effort,
            )
        from adaptive_shift_bench.codex_cli import CodexCLIAdapter

        return CodexCLIAdapter(
            model=model,
            workspace_dir=workspace_dir,
            effort=effort,
        )

    return LocalTaskLLM(factory)


def _load_tasks(suite: str, output_dir: Path):
    if suite == "v1":
        attempt_task, overall_task = build_kbench_tasks(output_dir=output_dir)
        return {"attempt": attempt_task, "overall": overall_task}
    if suite == "v2":
        attempt_task, sequence_task, overall_task = build_kbench_v2_tasks(output_dir=output_dir)
        return {"attempt": attempt_task, "sequence": sequence_task, "overall": overall_task}
    attempt_task, sequence_task, overall_task = build_kbench_v2_learning_tasks(output_dir=output_dir)
    return {"attempt": attempt_task, "sequence": sequence_task, "overall": overall_task}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kaggle task wrappers locally via a kaggle_benchmarks shim.")
    parser.add_argument("--backend", required=True, choices=("claude", "codex"))
    parser.add_argument("--suite", default="v2_learning", choices=("v1", "v2", "v2_learning"))
    parser.add_argument("--task", default="sequence", choices=("attempt", "sequence", "overall"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--workspace-dir", default="workspace/local_kbench_smoke")
    parser.add_argument("--scenario-id", action="append", dest="scenario_ids")
    parser.add_argument("--sequence-id", action="append", dest="sequence_ids")
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--max-budget-usd", type=float, default=3.00)
    args = parser.parse_args()

    workspace_root = Path(args.workspace_dir)
    workspace_root.mkdir(parents=True, exist_ok=True)

    with patched_local_kaggle_benchmarks():
        tasks = _load_tasks(args.suite, workspace_root / "reports")
        llm = _build_llm(args.backend, args.model, args.effort, workspace_root, args.max_budget_usd)

        if args.task == "overall":
            payload = {
                "suite": args.suite,
                "task": args.task,
                "score": tasks["overall"].run(llm=llm),
            }
        elif args.task == "attempt":
            if not args.scenario_ids:
                raise SystemExit("--scenario-id is required for --task attempt")
            cases = [
                {
                    "task": tasks["attempt"],
                    "kwargs": {"llm": llm, "scenario_id": scenario_id, "attempt_index": args.attempt_index},
                }
                for scenario_id in args.scenario_ids
            ]
            scores = run_parallel(cases, max_workers=args.parallelism) if len(cases) > 1 else [cases[0]["task"].run(**cases[0]["kwargs"])]
            payload = {
                "suite": args.suite,
                "task": args.task,
                "scores": dict(zip(args.scenario_ids, scores, strict=True)),
            }
        else:
            if args.suite == "v1":
                raise SystemExit("--task sequence is unavailable for --suite v1")
            if not args.sequence_ids:
                raise SystemExit("--sequence-id is required for --task sequence")
            cases = [
                {
                    "task": tasks["sequence"],
                    "kwargs": {"llm": llm, "sequence_id": sequence_id, "attempt_index": args.attempt_index},
                }
                for sequence_id in args.sequence_ids
            ]
            scores = run_parallel(cases, max_workers=args.parallelism) if len(cases) > 1 else [cases[0]["task"].run(**cases[0]["kwargs"])]
            payload = {
                "suite": args.suite,
                "task": args.task,
                "scores": dict(zip(args.sequence_ids, scores, strict=True)),
            }

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
