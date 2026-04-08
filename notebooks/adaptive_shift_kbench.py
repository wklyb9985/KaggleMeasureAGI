import subprocess
import sys
from shutil import which
import importlib

import kaggle_benchmarks as kbench


REPO_URL = "https://github.com/wklyb9985/KaggleMeasureAGI/archive/refs/heads/master.zip"
try:
    import pip  # noqa: F401
except ImportError:
    if which("uv"):
        subprocess.check_call(
            ["uv", "pip", "install", "--python", sys.executable, "-q", "--upgrade", "--reinstall", REPO_URL]
        )
    else:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall", "--upgrade", REPO_URL]
        )
else:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall", "--upgrade", REPO_URL]
    )

for module_name in [name for name in sys.modules if name == "adaptive_shift_bench" or name.startswith("adaptive_shift_bench.")]:
    del sys.modules[module_name]
importlib.invalidate_caches()

from adaptive_shift_bench.kaggle_tasks import get_public_kbench_v2_learning_tasks


(
    adaptive_shift_v2_learning_openai,
    adaptive_shift_v2_learning_pandas,
    adaptive_shift_v2_learning_registry,
    adaptive_shift_v2_learning_overall,
) = get_public_kbench_v2_learning_tasks()

# Run the canonical learning leaderboard task.
adaptive_shift_v2_learning_overall.run(llm=kbench.llm)

# Kaggle leaderboards support a single selected task per notebook.
# Explicitly choose the overall task so subtasks remain visible as breakdowns
# instead of being treated as the notebook's main benchmark output.
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("choose", "adaptive_shift_v2_learning_overall")
except Exception:
    pass

# Example sequence smoke:
# adaptive_shift_v2_learning_openai.run(llm=kbench.llm, attempt_index=0)
# adaptive_shift_v2_learning_pandas.run(llm=kbench.llm, attempt_index=0)
# adaptive_shift_v2_learning_registry.run(llm=kbench.llm, attempt_index=0)
