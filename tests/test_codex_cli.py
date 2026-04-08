from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from adaptive_shift_bench.codex_cli import CodexCLIAdapter


class CodexCLITests(unittest.TestCase):
    def test_codex_adapter_parses_jsonl_payload(self):
        stdout = "\n".join(
            [
                '{"type":"thread.started","thread_id":"thread-123"}',
                'not json',
                '{"type":"item.completed","item":{"type":"agent_message","text":"{\\"action\\":\\"answer\\",\\"content\\":\\"x\\"}"}}',
                '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":4,"output_tokens":2}}',
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = CodexCLIAdapter(model="gpt-5.4-mini", workspace_dir=temp_dir)
            with mock.patch("subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = stdout
                run.return_value.stderr = ""
                response = adapter.prompt("hello")

        self.assertEqual(response, '{"action":"answer","content":"x"}')
        self.assertEqual(len(adapter.invocations), 1)
        self.assertEqual(adapter.invocations[0].input_tokens, 10)
        self.assertEqual(adapter.invocations[0].cached_input_tokens, 4)
        self.assertEqual(adapter.invocations[0].output_tokens, 2)
        self.assertEqual(adapter.invocations[0].thread_id, "thread-123")


if __name__ == "__main__":
    unittest.main()
