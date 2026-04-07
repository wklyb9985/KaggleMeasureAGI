from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

from adaptive_shift_bench.claude_cli import ClaudeCLIAdapter, build_transcript_prompt
from adaptive_shift_bench.claude_runner import build_pilot_suite


class ClaudeCLITests(unittest.TestCase):
    def test_build_transcript_prompt_includes_history(self):
        prompt = build_transcript_prompt(
            [("user 1", "assistant 1"), ("user 2", "assistant 2")],
            "current",
        )
        self.assertIn("<turn index=\"1\">", prompt)
        self.assertIn("assistant 2", prompt)
        self.assertIn("<current_turn>", prompt)
        self.assertIn("current", prompt)

    def test_claude_adapter_parses_json_payload(self):
        payload = {
            "result": '{"action":"answer","content":"x"}',
            "duration_ms": 12,
            "total_cost_usd": 0.01,
            "modelUsage": {"claude-haiku-4-5-20251001": {"costUSD": 0.01}},
            "session_id": "abc",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = ClaudeCLIAdapter(model="haiku", workspace_dir=temp_dir)
            with mock.patch("subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = json.dumps(payload)
                run.return_value.stderr = ""
                response = adapter.prompt("hello")

        self.assertEqual(response, payload["result"])
        self.assertEqual(len(adapter.invocations), 1)
        self.assertEqual(adapter.invocations[0].total_cost_usd, 0.01)

    def test_pilot_suite_contains_representative_subset(self):
        suite = build_pilot_suite()
        self.assertEqual(len(suite), 6)
        self.assertEqual({scenario.family.value for scenario in suite}, {"api_migration", "dsl_wrapper", "future_registry"})


if __name__ == "__main__":
    unittest.main()
