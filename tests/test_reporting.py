from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_shift_bench.models import ActionRecord, EpisodeAction, EpisodeResult, ScenarioFamily, SequenceResult
from adaptive_shift_bench.reporting import (
    aggregate_attempts,
    aggregate_sequence_results,
    write_report_bundle,
    write_sequence_report_bundle,
)


def _result(scenario_id: str, family: ScenarioFamily, attempt_index: int, passed: bool, score: float) -> EpisodeResult:
    return EpisodeResult(
        scenario_id=scenario_id,
        family=family,
        attempt_index=attempt_index,
        passed=passed,
        score=score,
        failure_tags=(),
        selected_entities=(),
        used_evidence_ids=(),
        trace_digest=f"{scenario_id}-{attempt_index}",
        action_records=(ActionRecord(turn_index=0, action=EpisodeAction(action="answer"), observation="done"),),
        functional_correctness=passed,
        adaptation_correctness=passed,
        legacy_avoidance=passed,
        semantic_correctness=float(passed),
    )


def _sequence_result(
    sequence_id: str,
    family: ScenarioFamily,
    attempt_index: int,
    passed: bool,
    overall_score: float,
    *,
    semantic: float,
    adaptation: float,
    transfer: float,
    protocol: float,
    efficiency: float,
    benchmark_suite: str = "v2",
    prior_probe: float = 0.0,
    stale_prior_rate: float = 0.0,
    revision_success: float = 0.0,
    revision_turns_to_fix: float = 0.0,
    revision_efficiency: float = 0.0,
    transfer_after_revision: float = 0.0,
    localized_generalization: float = 0.0,
    learning_score: float = 0.0,
) -> SequenceResult:
    return SequenceResult(
        sequence_id=sequence_id,
        family=family,
        attempt_index=attempt_index,
        passed=passed,
        overall_score=overall_score,
        stage_results=(),
        in_task_adaptation=adaptation,
        cross_task_transfer=transfer,
        semantic_correctness=semantic,
        protocol_compliance=protocol,
        efficiency_score=efficiency,
        benchmark_suite=benchmark_suite,
        prior_probe_correctness=prior_probe,
        stale_prior_rate=stale_prior_rate,
        revision_success=revision_success,
        revision_turns_to_fix=revision_turns_to_fix,
        revision_efficiency=revision_efficiency,
        transfer_after_revision=transfer_after_revision,
        localized_generalization=localized_generalization,
        learning_score=learning_score,
    )


class ReportingTests(unittest.TestCase):
    def test_aggregate_attempts_computes_primary_metrics(self):
        results = [
            _result("s1", ScenarioFamily.API_MIGRATION, 0, True, 1.0),
            _result("s1", ScenarioFamily.API_MIGRATION, 1, False, 0.6),
            _result("s1", ScenarioFamily.API_MIGRATION, 2, False, 0.6),
            _result("s1", ScenarioFamily.API_MIGRATION, 3, False, 0.6),
            _result("s1", ScenarioFamily.API_MIGRATION, 4, False, 0.6),
            _result("s2", ScenarioFamily.FUTURE_REGISTRY, 0, False, 0.6),
            _result("s2", ScenarioFamily.FUTURE_REGISTRY, 1, True, 1.0),
            _result("s2", ScenarioFamily.FUTURE_REGISTRY, 2, False, 0.6),
            _result("s2", ScenarioFamily.FUTURE_REGISTRY, 3, False, 0.6),
            _result("s2", ScenarioFamily.FUTURE_REGISTRY, 4, False, 0.6),
        ]

        report = aggregate_attempts(results)

        self.assertAlmostEqual(report["metrics"]["pass_at_1"], 0.5)
        self.assertAlmostEqual(report["metrics"]["pass_at_5"], 1.0)
        self.assertAlmostEqual(report["metrics"]["avg5"], 0.68)
        self.assertAlmostEqual(report["metrics"]["overall"], 0.686)
        self.assertEqual(report["benchmark_metadata"]["feedback_only_policy"], "trial_first_reveal_after_fail")

    def test_aggregate_sequence_results_emits_v2_subscores(self):
        results = [
            _sequence_result(
                "seq-1",
                ScenarioFamily.API_MIGRATION,
                0,
                True,
                0.92,
                semantic=1.0,
                adaptation=1.0,
                transfer=0.8,
                protocol=0.9,
                efficiency=0.8,
            ),
            _sequence_result(
                "seq-1",
                ScenarioFamily.API_MIGRATION,
                1,
                False,
                0.60,
                semantic=0.7,
                adaptation=0.5,
                transfer=0.4,
                protocol=0.8,
                efficiency=0.9,
            ),
            _sequence_result(
                "seq-2",
                ScenarioFamily.DSL_WRAPPER,
                0,
                False,
                0.51,
                semantic=0.6,
                adaptation=0.4,
                transfer=0.2,
                protocol=0.7,
                efficiency=0.9,
            ),
            _sequence_result(
                "seq-2",
                ScenarioFamily.DSL_WRAPPER,
                1,
                True,
                0.95,
                semantic=1.0,
                adaptation=1.0,
                transfer=1.0,
                protocol=0.9,
                efficiency=0.8,
            ),
        ]

        report = aggregate_sequence_results(results)

        self.assertAlmostEqual(report["metrics"]["pass_at_1"], 0.5)
        self.assertAlmostEqual(report["metrics"]["pass_at_5"], 1.0)
        self.assertAlmostEqual(report["metrics"]["semantic_correctness"], 0.8)
        self.assertAlmostEqual(report["metrics"]["in_task_adaptation"], 0.7)
        self.assertAlmostEqual(report["metrics"]["cross_task_transfer"], 0.5)
        self.assertAlmostEqual(report["metrics"]["protocol_compliance"], 0.8)
        self.assertAlmostEqual(report["metrics"]["efficiency"], 0.85)
        self.assertEqual(report["benchmark_metadata"]["report_mode"], "v2_sequences")
        self.assertEqual(report["benchmark_metadata"]["benchmark_version"], "2.1")
        self.assertEqual(report["benchmark_metadata"]["comparability"], "not-comparable-to-pre-fix-v2")

    def test_write_report_bundle_outputs_json_and_markdown(self):
        report = aggregate_attempts(
            [
                _result("s1", ScenarioFamily.API_MIGRATION, 0, True, 1.0),
                _result("s1", ScenarioFamily.API_MIGRATION, 1, True, 1.0),
                _result("s1", ScenarioFamily.API_MIGRATION, 2, True, 1.0),
                _result("s1", ScenarioFamily.API_MIGRATION, 3, True, 1.0),
                _result("s1", ScenarioFamily.API_MIGRATION, 4, True, 1.0),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, md_path = write_report_bundle(report, temp_dir)
            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(md_path).exists())

    def test_write_sequence_report_bundle_outputs_json_and_markdown(self):
        report = aggregate_sequence_results(
            [
                _sequence_result(
                    "seq-1",
                    ScenarioFamily.API_MIGRATION,
                    0,
                    True,
                    0.95,
                    semantic=1.0,
                    adaptation=1.0,
                    transfer=1.0,
                    protocol=0.9,
                    efficiency=0.8,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, md_path = write_sequence_report_bundle(report, temp_dir)
            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(md_path).exists())
            markdown = Path(md_path).read_text(encoding="utf-8")
            self.assertIn("benchmark_version: 2.1", markdown)
            self.assertIn("comparability: not-comparable-to-pre-fix-v2", markdown)

    def test_aggregate_learning_sequence_results_emits_learning_metrics(self):
        report = aggregate_sequence_results(
            [
                _sequence_result(
                    "learn-1",
                    ScenarioFamily.API_MIGRATION,
                    0,
                    True,
                    0.72,
                    semantic=0.80,
                    adaptation=0.75,
                    transfer=0.50,
                    protocol=0.90,
                    efficiency=0.85,
                    benchmark_suite="v2_learning",
                    prior_probe=0.0,
                    stale_prior_rate=1.0,
                    revision_success=1.0,
                    revision_turns_to_fix=2.0,
                    revision_efficiency=1.0,
                    transfer_after_revision=1.0,
                    localized_generalization=1.0,
                    learning_score=1.0,
                ),
                _sequence_result(
                    "learn-2",
                    ScenarioFamily.DSL_WRAPPER,
                    0,
                    False,
                    0.40,
                    semantic=0.50,
                    adaptation=0.40,
                    transfer=0.20,
                    protocol=0.80,
                    efficiency=0.70,
                    benchmark_suite="v2_learning",
                    prior_probe=0.0,
                    stale_prior_rate=1.0,
                    revision_success=0.0,
                    revision_turns_to_fix=0.0,
                    revision_efficiency=0.0,
                    transfer_after_revision=0.0,
                    localized_generalization=0.0,
                    learning_score=0.0,
                ),
            ]
        )

        self.assertEqual(report["benchmark_metadata"]["benchmark_suite"], "v2_learning")
        self.assertEqual(report["benchmark_metadata"]["report_mode"], "v2_learning_sequences")
        self.assertEqual(report["benchmark_metadata"]["primary_leaderboard_metric"], "learning_score")
        self.assertEqual(report["benchmark_metadata"]["benchmark_version"], "2.3-learning-b")
        self.assertEqual(report["benchmark_metadata"]["learning_variant"], "b")
        self.assertAlmostEqual(report["metrics"]["prior_probe_correctness"], 0.0)
        self.assertAlmostEqual(report["metrics"]["stale_prior_rate"], 1.0)
        self.assertAlmostEqual(report["metrics"]["revision_success"], 0.5)
        self.assertAlmostEqual(report["metrics"]["transfer_after_revision"], 0.5)
        self.assertAlmostEqual(report["metrics"]["localized_generalization"], 0.5)
        self.assertAlmostEqual(report["metrics"]["learning_score"], 0.5)
        self.assertAlmostEqual(report["metrics"]["overall"], 0.5)

    def test_write_learning_sequence_report_bundle_uses_learning_filenames(self):
        report = aggregate_sequence_results(
            [
                _sequence_result(
                    "learn-1",
                    ScenarioFamily.API_MIGRATION,
                    0,
                    True,
                    0.72,
                    semantic=0.80,
                    adaptation=0.75,
                    transfer=0.50,
                    protocol=0.90,
                    efficiency=0.85,
                    benchmark_suite="v2_learning",
                    prior_probe=0.0,
                    stale_prior_rate=1.0,
                    revision_success=1.0,
                    revision_turns_to_fix=2.0,
                    revision_efficiency=1.0,
                    transfer_after_revision=1.0,
                    localized_generalization=1.0,
                    learning_score=1.0,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, md_path = write_sequence_report_bundle(report, temp_dir)
            self.assertEqual(Path(json_path).name, "adaptive_shift_v2_learning_report.json")
            self.assertEqual(Path(md_path).name, "adaptive_shift_v2_learning_report.md")
            markdown = Path(md_path).read_text(encoding="utf-8")
            self.assertIn("primary_leaderboard_metric: learning_score", markdown)
            self.assertIn("learning_score:", markdown)


    def test_transfer_and_adaptation_scores_differ(self):
        """in_task_adaptation and cross_task_transfer can have different values."""
        results = [
            _sequence_result(
                "seq-1",
                ScenarioFamily.API_MIGRATION,
                0,
                False,
                0.70,
                semantic=0.9,
                adaptation=1.0,
                transfer=0.2,
                protocol=0.9,
                efficiency=0.8,
            ),
        ]

        report = aggregate_sequence_results(results)

        self.assertNotAlmostEqual(
            report["metrics"]["in_task_adaptation"],
            report["metrics"]["cross_task_transfer"],
        )
        self.assertAlmostEqual(report["metrics"]["in_task_adaptation"], 1.0)
        self.assertAlmostEqual(report["metrics"]["cross_task_transfer"], 0.2)

    def test_family_breakdown_includes_all_subscores(self):
        """V2 family breakdown has all 5 component metrics."""
        results = [
            _sequence_result(
                "seq-1",
                ScenarioFamily.API_MIGRATION,
                0,
                True,
                0.92,
                semantic=1.0,
                adaptation=1.0,
                transfer=0.8,
                protocol=0.9,
                efficiency=0.8,
            ),
        ]

        report = aggregate_sequence_results(results)

        breakdown = report["family_breakdown"]
        self.assertIn("api_migration", breakdown)
        family = breakdown["api_migration"]
        for key in ("semantic_correctness", "in_task_adaptation", "cross_task_transfer", "protocol_compliance", "efficiency"):
            self.assertIn(key, family, f"Missing {key} in family breakdown")


if __name__ == "__main__":
    unittest.main()
