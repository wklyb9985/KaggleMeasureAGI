from __future__ import annotations

import unittest

from adaptive_shift_bench.engine import run_scenario
from adaptive_shift_bench.llm import ScriptedLLMAdapter
from adaptive_shift_bench.scenarios import build_core_suite, build_stress_suite, get_scenario


class EngineTests(unittest.TestCase):
    def test_suite_sizes_match_plan(self):
        self.assertEqual(len(build_core_suite()), 30)
        self.assertEqual(len(build_stress_suite()), 6)

    def test_docs_search_episode_uses_required_doc_and_passes(self):
        scenario = get_scenario("api_migration-easy-docs_search")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_docs","query":"openai unified responses create"}',
                '{"action":"read_doc","doc_id":"openai-unified-migration"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'ping\')"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertIn("openai-unified-migration", result.used_evidence_ids)
        self.assertTrue(result.adaptation_correctness)

    def test_feedback_only_registry_episode_can_recover_after_bad_selection(self):
        scenario = get_scenario("future_registry-hard-feedback_only")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"select_model","model":"gpt-8-mini"}',
                '{"action":"answer","content":"{\\"model\\": \\"gpt-7-mini-preview\\"}"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertIn("gpt-7-mini-preview", result.selected_entities)

    def test_raw_final_answer_fallback_is_accepted_when_correct(self):
        scenario = get_scenario("api_migration-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ["client.unified.responses.create(model='gpt-6-mini', input='ping')"]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertEqual(result.failure_tags, ())

    def test_structured_json_content_answer_is_accepted_for_registry_tasks(self):
        scenario = get_scenario("future_registry-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ['{"action":"answer","content":{"model":"gemini-5-flash"}}']
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertIn("gemini-5-flash", result.selected_entities)

    def test_release_note_prompt_style_changes_framing(self):
        scenario = get_scenario("api_migration-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ["client.unified.responses.create(model='gpt-6-mini', input='ping')"]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertTrue(adapter.seen_prompts)
        self.assertIn("I just released", adapter.seen_prompts[0])
        self.assertNotIn("offline adaptive interface benchmark", adapter.seen_prompts[0])


if __name__ == "__main__":
    unittest.main()
