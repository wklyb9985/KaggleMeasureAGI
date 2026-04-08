from __future__ import annotations

import ast
import importlib
import inspect
import os
import textwrap
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_shift_bench.kaggle_tasks import (
    build_kbench_tasks,
    build_kbench_v2_learning_tasks,
    build_kbench_v2_tasks,
    _report_attempt_path,
    _resolve_output_dir,
)
from adaptive_shift_bench.llm import ScriptedLLMAdapter
from adaptive_shift_bench.local_kaggle_mock import LocalTaskLLM, patched_local_kaggle_benchmarks, run_parallel


class KaggleTaskTests(unittest.TestCase):
    @staticmethod
    def _discover_subtask_names(task_obj) -> list[str]:
        source_code = textwrap.dedent(inspect.getsource(task_obj.func))
        tree = ast.parse(source_code)
        subtask_names = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
            ):
                continue
            var_expr = ast.unparse(node.func.value)
            if "." in var_expr or var_expr not in task_obj.func.__globals__:
                continue
            candidate = task_obj.func.__globals__[var_expr]
            if hasattr(candidate, "run") and getattr(candidate, "name", None) not in subtask_names:
                subtask_names.append(candidate.name)
        return sorted(subtask_names)

    def test_build_kbench_tasks_requires_kaggle_benchmarks(self):
        with self.assertRaises(ImportError):
            build_kbench_tasks()

    def test_build_kbench_v2_tasks_requires_kaggle_benchmarks(self):
        with self.assertRaises(ImportError):
            build_kbench_v2_tasks()

    def test_build_kbench_v2_learning_tasks_requires_kaggle_benchmarks(self):
        with self.assertRaises(ImportError):
            build_kbench_v2_learning_tasks()

    def test_v2_sequence_report_includes_learned_rules(self):
        """V2 sequence report JSON must contain learned_rules and stage_results."""
        import json as _json

        from adaptive_shift_bench.engine import run_sequence
        from adaptive_shift_bench.llm import ScriptedLLMAdapter
        from adaptive_shift_bench.scenarios import get_v2_sequence

        sequence = get_v2_sequence("v2-openai-unified")
        adapter = ScriptedLLMAdapter(
            [
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', input='ping')\"}",
                '{"action":"search_docs","query":"content field unified responses create"}',
                '{"action":"read_doc","doc_id":"openai-content-migration"}',
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', content='status')\"}",
                "{\"action\":\"run_candidate\",\"candidate\":\"client.unified.responses.create(model='gpt-6-mini', content='heartbeat')\"}",
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', content='heartbeat')\"}",
                '{"action":"search_docs","query":"中文 统一 响应 创建 模型 内容"}',
                '{"action":"read_doc","doc_id":"openai-han-overview"}',
                "{\"action\":\"answer\",\"content\":\"client.统一.响应.创建(模型='gpt-6-mini', 内容='ping')\"}",
            ]
        )
        result = run_sequence(adapter, sequence, attempt_index=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = _report_attempt_path("v2-sequence", 0, prefix="v2-sequence", output_dir=temp_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            from dataclasses import asdict
            report_data = asdict(result)
            path.write_text(_json.dumps(report_data, default=str))
            loaded = _json.loads(path.read_text())

        self.assertIn("learned_rules", loaded)
        self.assertIn("stage_results", loaded)
        self.assertIsInstance(loaded["learned_rules"], list)
        self.assertIsInstance(loaded["stage_results"], list)
        self.assertEqual(len(loaded["stage_results"]), 4)

    def test_report_attempt_path_is_namespaced_and_unique(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = _resolve_output_dir(temp_dir)
            first = _report_attempt_path("scenario", 0, output_dir=base)
            second = _report_attempt_path("scenario", 0, output_dir=base)
            sequence_path = _report_attempt_path("sequence", 0, prefix="v2-sequence", output_dir=base)

        self.assertTrue(str(first).startswith(str(Path(temp_dir))))
        self.assertNotEqual(first, second)
        self.assertIn("v2-sequence", sequence_path.name)

    def test_local_kaggle_mock_runs_learning_sequence_task(self):
        adapters = {}

        def factory(session_key: str):
            adapter = ScriptedLLMAdapter(
                [
                    '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'status\')"}',
                    '{"action":"search_docs","query":"latest text payload field unified responses content status"}',
                    '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'status\')"}',
                    '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
                    '{"action":"search_docs","query":"中文 接口 统一 响应 创建 模型 内容"}',
                    '{"action":"answer","content":"client.统一.响应.创建(模型=\'gpt-6-mini\', 内容=\'ping\')"}',
                ]
            )
            adapters[session_key] = adapter
            return adapter

        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_local_kaggle_benchmarks():
                _, sequence_task, _ = build_kbench_v2_learning_tasks(output_dir=temp_dir)
                llm = LocalTaskLLM(factory)
                score = sequence_task.run(llm=llm, sequence_id="v2-learning-openai-revision", attempt_index=0)

        self.assertEqual(score, 1.0)
        self.assertEqual(len(adapters), 1)
        adapter = next(iter(adapters.values()))
        self.assertGreater(len(adapter.seen_prompts), 1)

    def test_local_kaggle_mock_parallel_attempts_isolate_sessions(self):
        adapters = {}

        def factory(session_key: str):
            adapter = ScriptedLLMAdapter(
                ["client.unified.responses.create(model='gpt-6-mini', input='ping')"]
            )
            adapters[session_key] = adapter
            return adapter

        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_local_kaggle_benchmarks():
                attempt_task, _ = build_kbench_tasks(output_dir=temp_dir)
                llm = LocalTaskLLM(factory)
                scores = run_parallel(
                    [
                        {
                            "task": attempt_task,
                            "kwargs": {"llm": llm, "scenario_id": "api_migration-easy-explicit_change", "attempt_index": 0},
                        },
                        {
                            "task": attempt_task,
                            "kwargs": {"llm": llm, "scenario_id": "api_migration-easy-explicit_change", "attempt_index": 0},
                        },
                    ],
                    max_workers=2,
                )

        self.assertEqual(scores, [1.0, 1.0])
        self.assertEqual(len(adapters), 2)

    def test_local_kaggle_tasks_keep_runtime_float_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_local_kaggle_benchmarks():
                attempt_task, sequence_task, overall_task = build_kbench_v2_learning_tasks(output_dir=temp_dir)

        self.assertIs(attempt_task.func.__annotations__["return"], float)
        self.assertIs(sequence_task.func.__annotations__["return"], float)
        self.assertIs(overall_task.func.__annotations__["return"], float)

    def test_public_learning_tasks_expose_sequence_subtasks(self):
        with patched_local_kaggle_benchmarks():
            module = importlib.import_module("adaptive_shift_bench.kaggle_tasks")
            importlib.reload(module)
            _, _, _, overall_task = module.get_public_kbench_v2_learning_tasks()

        self.assertEqual(
            self._discover_subtask_names(overall_task),
            [
                "adaptive_shift_v2_learning_openai",
                "adaptive_shift_v2_learning_pandas",
                "adaptive_shift_v2_learning_registry",
            ],
        )

    def test_public_learning_openai_task_writes_deterministic_sequence_report(self):
        adapters = {}

        def factory(session_key: str):
            adapter = ScriptedLLMAdapter(
                [
                    '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'status\')"}',
                    '{"action":"search_docs","query":"latest text payload field unified responses content status"}',
                    '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'status\')"}',
                    '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
                    '{"action":"search_docs","query":"中文 接口 统一 响应 创建 模型 内容"}',
                    '{"action":"answer","content":"client.统一.响应.创建(模型=\'gpt-6-mini\', 内容=\'ping\')"}',
                ]
            )
            adapters[session_key] = adapter
            return adapter

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"ADAPTIVE_SHIFT_OUTPUT_DIR": temp_dir}, clear=False):
                with patched_local_kaggle_benchmarks():
                    module = importlib.import_module("adaptive_shift_bench.kaggle_tasks")
                    importlib.reload(module)
                    openai_task, _, _, _ = module.get_public_kbench_v2_learning_tasks()
                    llm = LocalTaskLLM(factory)
                    score = openai_task.run(llm=llm, attempt_index=0)
                    report_path = module._public_sequence_report_path("v2-learning-openai-revision", 0)
                    payload = report_path.read_text(encoding="utf-8")

        self.assertEqual(score, 1.0)
        self.assertTrue(payload)
        self.assertEqual(len(adapters), 1)


if __name__ == "__main__":
    unittest.main()
