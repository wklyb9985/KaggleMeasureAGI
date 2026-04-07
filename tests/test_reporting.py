from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adaptive_shift_bench.models import ActionRecord, EpisodeAction, EpisodeResult, ScenarioFamily
from adaptive_shift_bench.reporting import aggregate_attempts, write_report_bundle


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


if __name__ == "__main__":
    unittest.main()

