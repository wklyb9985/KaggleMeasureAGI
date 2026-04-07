import kaggle_benchmarks as kbench

from adaptive_shift_bench.kaggle_tasks import build_kbench_tasks


adaptive_shift_attempt, adaptive_shift_overall = build_kbench_tasks()

# Run the public leaderboard task.
adaptive_shift_overall.run(llm=kbench.llm)
