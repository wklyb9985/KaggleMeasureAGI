import subprocess
import sys

REPO_URL = "git+https://github.com/wklyb9985/KaggleMeasureAGI.git"
try:
    import pip  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", REPO_URL])

from adaptive_shift_bench.kaggle_tasks import build_kbench_v2_learning_tasks
from adaptive_shift_bench.local_kaggle_mock import LocalTaskLLM, patched_local_kaggle_benchmarks
from adaptive_shift_bench.llm import ScriptedLLMAdapter


def _adapter_factory(_: str) -> ScriptedLLMAdapter:
    return ScriptedLLMAdapter(
        [
            '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', input=\'status\')"}',
            '{"action":"search_docs","query":"latest text payload field unified responses content status"}',
            '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'status\')"}',
            '{"action":"answer","content":"client.unified.responses.create(model=\'gpt-6-mini\', content=\'heartbeat\')"}',
            '{"action":"search_docs","query":"中文 接口 统一 响应 创建 模型 内容"}',
            '{"action":"answer","content":"client.统一.响应.创建(模型=\'gpt-6-mini\', 内容=\'ping\')"}',
        ]
    )


with patched_local_kaggle_benchmarks():
    _, adaptive_shift_v2_learning_sequence, _ = build_kbench_v2_learning_tasks(
        output_dir="workspace/local_kbench_demo"
    )
    llm = LocalTaskLLM(_adapter_factory)

    score = adaptive_shift_v2_learning_sequence.run(
        llm=llm,
        sequence_id="v2-learning-openai-revision",
        attempt_index=0,
    )
    print({"sequence_id": "v2-learning-openai-revision", "learning_score": score})
