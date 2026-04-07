from __future__ import annotations

import unittest

from adaptive_shift_bench.kaggle_tasks import build_kbench_tasks


class KaggleTaskTests(unittest.TestCase):
    def test_build_kbench_tasks_requires_kaggle_benchmarks(self):
        with self.assertRaises(ImportError):
            build_kbench_tasks()


if __name__ == "__main__":
    unittest.main()
