import importlib
import subprocess
import sys
from shutil import which

import kaggle_benchmarks as kbench


REPO_URL = "https://github.com/wklyb9985/KaggleMeasureAGI/archive/28862a2b37a99ee40a79d9b75bf33d0206cd7be4.zip"
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

from adaptive_shift_bench.kaggle_tasks import get_public_kbench_v3_learning_strict_tasks


(
    adaptive_shift_v3_learning_strict_hard_abstract,
    adaptive_shift_v3_learning_strict_hard_realistic,
    adaptive_shift_v3_learning_strict_hard_overall,
) = get_public_kbench_v3_learning_strict_tasks("hard")

adaptive_shift_v3_learning_strict_hard_overall.run(llm=kbench.llm)

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("choose", "adaptive_shift_v3_learning_strict_hard_overall")
except Exception:
    pass
