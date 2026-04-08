import subprocess
import sys

import kaggle_benchmarks as kbench


REPO_URL = "git+https://github.com/wklyb9985/KaggleMeasureAGI.git"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", REPO_URL])

from adaptive_shift_bench.kaggle_tasks import build_kbench_v2_learning_tasks


(
    adaptive_shift_v2_learning_attempt,
    adaptive_shift_v2_learning_sequence,
    adaptive_shift_v2_learning_overall,
) = build_kbench_v2_learning_tasks()

# Run the canonical learning leaderboard task.
adaptive_shift_v2_learning_overall.run(llm=kbench.llm)

# Example sequence smoke:
# adaptive_shift_v2_learning_sequence.run(
#     llm=kbench.llm,
#     sequence_id="v2-learning-openai-revision",
#     attempt_index=0,
# )
