from __future__ import annotations

import unittest

from adaptive_shift_bench.engine import (
    _build_recording_env,
    _execute_python,
    _recorded_call_matches,
    run_scenario,
    run_sequence,
)
from adaptive_shift_bench.llm import ScriptedLLMAdapter
from adaptive_shift_bench.models import EvidenceRegime, LanguageSurface, ScenarioFamily
from adaptive_shift_bench.scenarios import (
    build_core_suite,
    build_v2_learning_variant_a_sequences,
    build_stress_suite,
    build_v2_learning_variant_c_sequences,
    build_v2_learning_sequences,
    build_v2_learning_stage_suite,
    build_v2_sequences,
    build_v2_stage_suite,
    get_scenario,
    get_v2_learning_scenario,
    get_v2_learning_sequence,
    get_v2_scenario,
    get_v2_sequence,
)


class EngineTests(unittest.TestCase):
    def test_suite_sizes_match_plan(self):
        self.assertEqual(len(build_core_suite()), 30)
        self.assertEqual(len(build_stress_suite()), 6)
        self.assertEqual(len(build_v2_sequences()), 3)
        self.assertEqual(len(build_v2_stage_suite()), 12)
        self.assertEqual(len(build_v2_learning_sequences()), 3)
        self.assertEqual(len(build_v2_learning_stage_suite()), 12)
        self.assertTrue(all(scenario.timeout_s == 600 for scenario in build_core_suite()))
        self.assertTrue(all(scenario.timeout_s == 600 for scenario in build_stress_suite()))
        for scenario in build_core_suite():
            if scenario.evidence_regime == EvidenceRegime.FEEDBACK_ONLY:
                self.assertEqual(scenario.max_turns, 15)
            else:
                self.assertEqual(scenario.max_turns, 4)

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
                '{"action":"answer","content":"{\\"model\\": \\\"gpt-7-mini-preview\\\"}"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertIn("gpt-7-mini-preview", result.selected_entities)

    def test_feedback_only_initial_prompt_hides_docs_and_answer(self):
        scenario = get_scenario("api_migration-hard-feedback_only")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"run_candidate","candidate":"torch.cuda.amp.autocast()"}',
                '{"action":"answer","content":"torch.amp.scope(device=\'cuda\', dtype=\'float16\')"}',
            ]
        )

        run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(adapter.seen_prompts)
        self.assertNotIn("search_local_docs", adapter.seen_prompts[0])
        self.assertNotIn("read_local_doc", adapter.seen_prompts[0])
        self.assertNotIn('{"action":"answer"', adapter.seen_prompts[0])

    def test_feedback_only_rejects_pretrial_doc_search(self):
        scenario = get_scenario("api_migration-hard-feedback_only")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_local_docs","query":"torch amp scope"}',
                '{"action":"run_candidate","candidate":"torch.cuda.amp.autocast()"}',
                '{"action":"answer","content":"torch.amp.scope(device=\'cuda\', dtype=\'float16\')"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertIn("Local docs are not available yet.", adapter.seen_prompts[1])
        self.assertNotIn("search_local_docs", adapter.seen_prompts[1])
        self.assertNotIn("read_local_doc", adapter.seen_prompts[1])

    def test_feedback_only_failed_trial_unlocks_docs(self):
        scenario = get_scenario("api_migration-hard-feedback_only")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"run_candidate","candidate":"torch.cuda.amp.autocast()"}',
                '{"action":"read_local_doc","doc_id":"torch-amp-scope-migration"}',
                '{"action":"answer","content":"torch.amp.scope(device=\'cuda\', dtype=\'float16\')"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertIn("Local docs are now available", adapter.seen_prompts[1])
        self.assertIn("search_local_docs", adapter.seen_prompts[1])
        self.assertIn("read_local_doc", adapter.seen_prompts[1])
        self.assertIn('{"action":"answer"', adapter.seen_prompts[1])

    def test_feedback_only_successful_first_trial_unlocks_answer_without_docs(self):
        scenario = get_scenario("future_registry-easy-feedback_only")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"select_model","model":"gemini-5-flash"}',
                '{"action":"answer","content":"{\\"model\\": \\\"gemini-5-flash\\\"}"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertIn('{"action":"answer"', adapter.seen_prompts[1])
        self.assertNotIn("search_local_docs", adapter.seen_prompts[1])
        self.assertNotIn("read_local_doc", adapter.seen_prompts[1])

    def test_semantic_validator_accepts_wrapped_python_answer(self):
        scenario = get_scenario("api_migration-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            [
                "{\"action\":\"answer\",\"content\":\"response = client.unified.responses.create(model='gpt-6-mini', input='ping')\\nprint(response)\"}"
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertEqual(result.failure_tags, ())

    def test_registry_final_answer_requires_structured_json(self):
        scenario = get_scenario("future_registry-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"answer","content":"The fastest available local Gemini model is gemini-5-flash."}'
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertFalse(result.passed)
        self.assertIn("model_parse_error", result.failure_tags)

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

    def test_registry_bare_json_final_answer_is_accepted(self):
        scenario = get_scenario("future_registry-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ['{"model":"gemini-5-flash"}']
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertTrue(result.passed)
        self.assertIn("gemini-5-flash", result.selected_entities)

    def test_localized_registry_bare_json_final_answer_is_accepted(self):
        scenario = get_v2_scenario("v2-localized-registry-capstone")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_docs","query":"gpt-8-mini 回退 模型"}',
                '{"模型":"gpt-7-mini-preview"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertIn("openai-han-registry", result.surfaced_evidence_ids)

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

    def test_v2_sequences_end_with_localized_capstone(self):
        sequences = build_v2_sequences()

        self.assertEqual({sequence.family for sequence in sequences}, {
            ScenarioFamily.API_MIGRATION,
            ScenarioFamily.DSL_WRAPPER,
            ScenarioFamily.FUTURE_REGISTRY,
        })
        for sequence in sequences:
            self.assertEqual(len(sequence.stages), 4)
            self.assertEqual(sequence.stages[-1].family, ScenarioFamily.LOCALIZED_API)
            self.assertEqual(sequence.stages[-1].language_surface, LanguageSurface.CHINESE)

    def test_v2_learning_sequences_have_probe_revision_transfer_generalization(self):
        sequences = build_v2_learning_sequences()

        self.assertEqual(len(sequences), 3)
        for sequence in sequences:
            self.assertEqual(sequence.benchmark_suite, "v2_learning")
            self.assertEqual(len(sequence.stages), 4)
            self.assertEqual(sequence.stages[0].sequence_stage.value, "teach")
            self.assertEqual(sequence.stages[1].sequence_stage.value, "adapt")
            self.assertEqual(sequence.stages[2].sequence_stage.value, "transfer")
            self.assertEqual(sequence.stages[3].sequence_stage.value, "capstone")

    def test_v2_learning_variant_c_is_accessible_without_changing_default_suite(self):
        self.assertEqual(len(build_v2_learning_sequences()), 3)
        variant_sequences = build_v2_learning_variant_c_sequences()
        self.assertEqual(len(variant_sequences), 1)
        sequence = get_v2_learning_sequence("v2-learning-registry-revision-c")
        self.assertEqual(sequence.id, "v2-learning-registry-revision-c")
        self.assertEqual(sequence.family, ScenarioFamily.FUTURE_REGISTRY)
        self.assertIn("fictional company founded in 2026", sequence.stages[0].prompt)
        self.assertIn("small, lite, and mini", sequence.stages[1].docs_index[0].text)
        self.assertEqual(
            [stage.id for stage in sequence.stages],
            [
                "v2-learning-registry-c-prior-probe",
                "v2-learning-registry-c-revision",
                "v2-learning-registry-c-transfer",
                "v2-learning-registry-c-localized-generalization",
            ],
        )

    def test_v2_learning_registry_defaults_to_variant_b(self):
        sequence = get_v2_learning_sequence("v2-learning-registry-revision")

        self.assertEqual(sequence.family, ScenarioFamily.FUTURE_REGISTRY)
        self.assertIn("Z.ai", sequence.stages[0].prompt)
        self.assertIn("glm-8-small", sequence.stages[0].prompt)
        self.assertIn("glm-4.5-air", sequence.stages[1].prompt)
        self.assertEqual(sequence.stages[1].docs_index[2].doc_id, "registry-learning-fallback-example")

    def test_v2_learning_variant_a_is_archived_but_accessible(self):
        variant_sequences = build_v2_learning_variant_a_sequences()

        self.assertEqual(len(variant_sequences), 1)
        sequence = get_v2_learning_sequence("v2-learning-registry-revision-a")
        self.assertEqual(sequence.id, "v2-learning-registry-revision-a")
        self.assertIn("OpenAI", sequence.stages[0].prompt)
        self.assertIn("gpt-8-small", sequence.stages[0].prompt)

    def test_v2_sequence_scores_cross_task_transfer(self):
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

        self.assertTrue(result.passed)
        self.assertEqual(len(result.stage_results), 4)
        self.assertAlmostEqual(result.cross_task_transfer, 1.0)
        self.assertAlmostEqual(result.semantic_correctness, 1.0)
        self.assertIn("openai.content_field", result.learned_rules)

    def test_v2_learning_sequence_scores_revision_after_wrong_prior(self):
        sequence = get_v2_learning_sequence("v2-learning-openai-revision")
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

        result = run_sequence(adapter, sequence, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertEqual(result.benchmark_suite, "v2_learning")
        self.assertAlmostEqual(result.prior_probe_correctness, 0.0)
        self.assertAlmostEqual(result.stale_prior_rate, 1.0)
        self.assertAlmostEqual(result.revision_success, 1.0)
        self.assertAlmostEqual(result.transfer_after_revision, 1.0)
        self.assertAlmostEqual(result.localized_generalization, 1.0)
        self.assertAlmostEqual(result.learning_score, 1.0)
        self.assertAlmostEqual(result.overall_score, 1.0)
        self.assertIn("stale_prior_guess", result.stage_results[0].failure_tags)

    def test_v2_learning_revision_stage_exposes_runnable_example_docs(self):
        openai_sequence = get_v2_learning_sequence("v2-learning-openai-revision")
        pandas_sequence = get_v2_learning_sequence("v2-learning-pandas-revision")
        registry_sequence = get_v2_learning_sequence("v2-learning-registry-revision")

        self.assertEqual(openai_sequence.stages[1].docs_index[2].doc_id, "openai-learning-content-example")
        self.assertIn("Runnable local example", openai_sequence.stages[1].docs_index[2].text)
        self.assertEqual(pandas_sequence.stages[1].docs_index[2].doc_id, "pandas-learning-stack-example")
        self.assertIn("orders_a", pandas_sequence.stages[1].docs_index[2].text)
        self.assertEqual(registry_sequence.stages[1].docs_index[2].doc_id, "registry-learning-fallback-example")
        self.assertNotIn("glm-8-lite", registry_sequence.stages[1].docs_index[2].text)

    def test_v2_learning_revision_tags_persisted_stale_prior(self):
        sequence = get_v2_learning_sequence("v2-learning-openai-revision")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'status\')"}',
                '{"action":"search_docs","query":"latest text payload field unified responses content status"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'status\')"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'heartbeat\')"}',
                '{"action":"search_docs","query":"中文 接口 统一 响应 创建 模型 内容"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'ping\')"}',
            ]
        )

        result = run_sequence(adapter, sequence, attempt_index=0, prompt_style="release_note")

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.revision_success, 0.0)
        self.assertIn("searched_but_not_updated", result.stage_results[1].failure_tags)

    def test_environment_rejected_tag_marks_refusal_without_evidence(self):
        sequence = get_v2_learning_sequence("v2-learning-registry-revision")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"answer","content":"{\\"model\\": \\"glm-4.5-air\\"}"}',
                '{"action":"answer","content":"I have no local registry docs in this workspace and no tools to search local files. Please provide the local registry document directly."}',
                '{"action":"answer","content":"I have no local registry docs in this workspace. Please provide the registry document directly."}',
                '{"action":"answer","content":"I have no local registry docs in this workspace. Please provide the registry document directly."}',
            ]
        )

        result = run_sequence(adapter, sequence, attempt_index=0, prompt_style="release_note")

        self.assertIn("environment_rejected", result.stage_results[1].failure_tags)

    def test_v2_docs_search_can_pass_from_search_results_only(self):
        scenario = get_v2_scenario("v2-api-openai-adapt")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_docs","query":"newest request shape Responses API gpt-6-mini content status"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'status\')"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertIn("openai-content-migration", result.surfaced_evidence_ids)

    def test_v2_docs_search_accepts_noncanonical_read(self):
        scenario = get_v2_scenario("v2-api-openai-adapt")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_docs","query":"request example status content unified responses"}',
                '{"action":"read_doc","doc_id":"openai-content-example"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'status\')"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertIn("openai-content-example", result.used_evidence_ids)


    # --- Recording mock tests ---

    def test_recording_mock_captures_correct_call(self):
        """Recording proxy captures the correct call path and kwargs."""
        env, log = _build_recording_env("openai_client")
        client = env["client"]
        client.unified.responses.create(model="gpt-6-mini", input="ping")

        self.assertEqual(len(log.calls), 1)
        call = log.calls[0]
        self.assertEqual(call.path, "client.unified.responses.create")
        self.assertEqual(call.kwargs, {"model": "gpt-6-mini", "input": "ping"})
        self.assertFalse(call.raised)

    def test_recording_mock_captures_forbidden_call_even_when_caught(self):
        """A deprecated call inside try/except is still recorded with raised=True."""
        env, log = _build_recording_env("openai_client")
        client = env["client"]
        try:
            client.responses.create(model="gpt-6-mini", input="ping")
        except Exception:
            pass

        self.assertEqual(len(log.calls), 1)
        call = log.calls[0]
        self.assertEqual(call.path, "client.responses.create")
        self.assertTrue(call.raised)

    def test_wrapped_python_passes_via_recording(self):
        """Code with assignment + print passes because recording captures the correct call."""
        ok, _, log = _execute_python(
            "response = client.unified.responses.create(model='gpt-6-mini', input='ping')\nprint(response)",
            "openai_client",
        )
        self.assertTrue(ok)
        successful = [c for c in log.calls if not c.raised]
        self.assertTrue(
            any(
                _recorded_call_matches(c, {"path": "client.unified.responses.create", "kwargs": {"model": "gpt-6-mini", "input": "ping"}})
                for c in successful
            )
        )

    def test_v2_pandas_teach_accepts_import_alias(self):
        """Normal import style should not fail the pandas V2 teach stage."""
        scenario = get_v2_scenario("v2-dsl-pandas-teach")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"answer","content":"import pandas_cn as pd\\nresult = pd.concat_rows([left, right], ignore_index=True)"}'
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertEqual(result.failure_tags, ())

    def test_code_with_only_forbidden_call_fails(self):
        """Code that only calls the deprecated path gets runtime_error and the call is recorded."""
        ok, feedback, log = _execute_python(
            "client.responses.create(model='gpt-6-mini', input='ping')",
            "openai_client",
        )
        self.assertFalse(ok)
        self.assertIn("runtime error", feedback)
        self.assertTrue(any(c.raised for c in log.calls))

    # --- Semantic / protocol split tests ---

    def test_registry_correct_model_wrong_format_gets_semantic_credit(self):
        """Prose answer with correct model name gets semantic_correctness > 0 but passed=False."""
        scenario = get_scenario("future_registry-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ['{"action":"answer","content":"gemini-5-flash"}']
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertFalse(result.passed)
        self.assertIn("model_parse_error", result.failure_tags)
        self.assertAlmostEqual(result.semantic_correctness, 1.0)

    def test_python_mock_correct_call_with_legacy_gets_semantic_credit(self):
        """Code with both correct and deprecated calls: semantic_correctness=1.0 but legacy_avoidance=False."""
        import json as _json

        scenario = get_scenario("api_migration-easy-explicit_change")
        code = (
            "try:\n"
            "    client.responses.create(model='gpt-6-mini', input='ping')\n"
            "except Exception:\n"
            "    pass\n"
            "client.unified.responses.create(model='gpt-6-mini', input='ping')\n"
        )
        adapter = ScriptedLLMAdapter(
            [_json.dumps({"action": "answer", "content": code})]
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertAlmostEqual(result.semantic_correctness, 1.0)
        self.assertFalse(result.legacy_avoidance)

    # --- Registry validation tests ---

    def test_registry_mid_episode_rejects_prose(self):
        """select_model with prose sentence fails (spaces rejected by tightened raw fallback)."""
        scenario = get_scenario("future_registry-easy-feedback_only")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"select_model","model":"The best model is gemini-5-flash"}',
                '{"action":"answer","content":"{\\"model\\": \\\"gemini-5-flash\\\"}"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertIn("model_parse_error", result.failure_tags)

    def test_registry_rejects_empty_model(self):
        """JSON with empty model value is rejected."""
        scenario = get_scenario("future_registry-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ['{"action":"answer","content":"{\\"model\\": \\\"\\\"}"}']
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertFalse(result.passed)
        self.assertIn("model_parse_error", result.failure_tags)

    def test_registry_rejects_extra_fields(self):
        """JSON with extra fields is rejected."""
        scenario = get_scenario("future_registry-easy-explicit_change")
        adapter = ScriptedLLMAdapter(
            ['{"action":"answer","content":"{\\"model\\": \\"gemini-5-flash\\", \\"reason\\": \\"fastest\\"}"}']
        )

        result = run_scenario(adapter, scenario, attempt_index=0)

        self.assertFalse(result.passed)
        self.assertIn("model_parse_error", result.failure_tags)

    # --- Sequence isolation tests ---

    def test_v2_sequence_failed_teach_yields_zero_transfer(self):
        """If TEACH fails, TRANSFER gets learning_transfer_score=0.0."""
        sequence = get_v2_sequence("v2-openai-unified")
        adapter = ScriptedLLMAdapter(
            [
                # TEACH: submit WRONG answer (legacy API)
                '{"action":"answer","content":"client.responses.create(model=\'gpt-6-mini\', input=\'ping\')"}',
                # ADAPT: also wrong
                '{"action":"answer","content":"client.responses.create(model=\'gpt-6-mini\', content=\'status\')"}',
                # TRANSFER: wrong (no rules learned)
                '{"action":"run_candidate","candidate":"client.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
                '{"action":"answer","content":"client.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
                # CAPSTONE: wrong
                '{"action":"answer","content":"client.responses.create(model=\'gpt-6-mini\', input=\'ping\')"}',
            ]
        )

        result = run_sequence(adapter, sequence, attempt_index=0)

        self.assertAlmostEqual(result.cross_task_transfer, 0.0)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.learned_rules), 0)

    def test_v2_transfer_stage_never_unlocks_docs(self):
        """Transfer-only stages reject doc access before and after a failed trial."""
        scenario = get_v2_scenario("v2-api-openai-transfer")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_local_docs","query":"openai content field"}',
                '{"action":"run_candidate","candidate":"client.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
                '{"action":"read_local_doc","doc_id":"openai-content-migration"}',
                '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
            ]
        )

        result = run_scenario(adapter, scenario, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertNotIn("search_local_docs", adapter.seen_prompts[0])
        self.assertNotIn("read_local_doc", adapter.seen_prompts[0])
        self.assertIn("Local docs are not available in this transfer stage.", adapter.seen_prompts[1])
        self.assertIn("Local docs are not available in this transfer stage.", adapter.seen_prompts[3])

    def test_v2_sequence_semantic_without_evidence_does_not_teach_rule(self):
        """A semantically correct stage without adaptation evidence does not teach transfer rules."""
        sequence = get_v2_sequence("v2-openai-unified")
        adapter = ScriptedLLMAdapter(
            [
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', input='ping')\"}",
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', content='status')\"}",
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', content='heartbeat')\"}",
                "{\"action\":\"answer\",\"content\":\"client.统一.响应.创建(模型='gpt-6-mini', 内容='ping')\"}",
            ]
        )

        result = run_sequence(adapter, sequence, attempt_index=0)

        self.assertIn("openai.unified_path", result.learned_rules)
        self.assertNotIn("openai.content_field", result.learned_rules)
        self.assertAlmostEqual(result.cross_task_transfer, 1.0 / 6.0)

    def test_v2_sequence_no_cross_sequence_leakage(self):
        """Two independent sequence runs produce consistent results with no state leakage."""
        sequence = get_v2_sequence("v2-openai-unified")
        responses = [
            # TEACH: correct
            "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', input='ping')\"}",
            # ADAPT: correct
            '{"action":"search_docs","query":"content field unified responses create"}',
            '{"action":"read_doc","doc_id":"openai-content-migration"}',
            "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', content='status')\"}",
            # TRANSFER: correct
            "{\"action\":\"run_candidate\",\"candidate\":\"client.unified.responses.create(model='gpt-6-mini', content='heartbeat')\"}",
            "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', content='heartbeat')\"}",
            # CAPSTONE: correct
            '{"action":"search_docs","query":"中文 统一 响应 创建 模型 内容"}',
            '{"action":"read_doc","doc_id":"openai-han-overview"}',
            "{\"action\":\"answer\",\"content\":\"client.统一.响应.创建(模型='gpt-6-mini', 内容='ping')\"}",
        ]
        adapter1 = ScriptedLLMAdapter(list(responses))
        result1 = run_sequence(adapter1, sequence, attempt_index=0)

        adapter2 = ScriptedLLMAdapter(list(responses))
        result2 = run_sequence(adapter2, sequence, attempt_index=0)

        self.assertEqual(result1.passed, result2.passed)
        self.assertAlmostEqual(result1.overall_score, result2.overall_score)
        self.assertAlmostEqual(result1.cross_task_transfer, result2.cross_task_transfer)

    # --- Localized hard-tier tests ---

    def test_localized_capstone_chinese_call_passes(self):
        """Chinese translated identifiers pass the localized capstone scenario."""
        capstone = get_v2_scenario("v2-localized-openai-capstone")
        adapter = ScriptedLLMAdapter(
            [
                '{"action":"search_docs","query":"统一 响应 创建 模型 内容"}',
                '{"action":"read_doc","doc_id":"openai-han-overview"}',
                "{\"action\":\"answer\",\"content\":\"client.统一.响应.创建(模型='gpt-6-mini', 内容='ping')\"}",
            ]
        )

        result = run_scenario(adapter, capstone, attempt_index=0, prompt_style="release_note")

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.semantic_correctness, 1.0)

    def test_localized_capstone_english_call_fails(self):
        """English identifiers for a Chinese capstone fail (forbidden call path)."""
        capstone = get_v2_scenario("v2-localized-openai-capstone")
        adapter = ScriptedLLMAdapter(
            [
                "{\"action\":\"answer\",\"content\":\"client.unified.responses.create(model='gpt-6-mini', input='ping')\"}"
            ]
        )

        result = run_scenario(adapter, capstone, attempt_index=0, prompt_style="release_note")

        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
