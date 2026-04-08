from __future__ import annotations

import json

from adaptive_shift_bench.scenarios import (
    build_core_suite,
    build_stress_suite,
    build_v2_learning_variant_a_sequences,
    build_v2_learning_variant_c_sequences,
    build_v2_learning_sequences,
    build_v2_learning_stage_suite,
    build_v2_sequences,
    build_v2_stage_suite,
)


def main() -> None:
    payload = {
        "core_suite_size": len(build_core_suite()),
        "stress_suite_size": len(build_stress_suite()),
        "v2_sequence_count": len(build_v2_sequences()),
        "v2_stage_count": len(build_v2_stage_suite()),
        "v2_learning_sequence_count": len(build_v2_learning_sequences()),
        "v2_learning_stage_count": len(build_v2_learning_stage_suite()),
        "v2_learning_archived_sequence_count": len(build_v2_learning_variant_a_sequences()),
        "v2_learning_experimental_sequence_count": len(build_v2_learning_variant_c_sequences()),
        "core_scenarios": [scenario.id for scenario in build_core_suite()],
        "v2_sequences": [sequence.id for sequence in build_v2_sequences()],
        "v2_learning_sequences": [sequence.id for sequence in build_v2_learning_sequences()],
        "v2_learning_archived_sequences": [
            sequence.id for sequence in build_v2_learning_variant_a_sequences()
        ],
        "v2_learning_experimental_sequences": [
            sequence.id for sequence in build_v2_learning_variant_c_sequences()
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
