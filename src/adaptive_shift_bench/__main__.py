from __future__ import annotations

import json

from adaptive_shift_bench.scenarios import build_core_suite, build_stress_suite


def main() -> None:
    payload = {
        "core_suite_size": len(build_core_suite()),
        "stress_suite_size": len(build_stress_suite()),
        "core_scenarios": [scenario.id for scenario in build_core_suite()],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

