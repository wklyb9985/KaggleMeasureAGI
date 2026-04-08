from __future__ import annotations

from functools import lru_cache

from adaptive_shift_bench.models import (
    Document,
    EvidenceRegime,
    LanguageSurface,
    Scenario,
    ScenarioFamily,
    ScenarioSequence,
    SequenceStage,
    ValidatorKind,
    ValidatorSpec,
)

DEFAULT_ATTEMPTS = 5
DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_TURNS = 4
FEEDBACK_ONLY_MAX_TURNS = 15
TRANSFER_ONLY_MAX_TURNS = 6


def _doc(doc_id: str, title: str, text: str, *keywords: str) -> Document:
    return Document(doc_id=doc_id, title=title, text=text.strip(), keywords=keywords)


def _name(identifier: str) -> str:
    return f"$name:{identifier}"


def _api_world(difficulty: str) -> dict:
    if difficulty == "easy":
        return {
            "base_id": "openai_unified",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `ping` to model `gpt-6-mini` with the latest SDK path."
            ),
            "required": (),
            "forbidden": ("client.responses.create",),
            "docs": (
                _doc(
                    "openai-unified-migration",
                    "OpenAI SDK 3.1 migration guide",
                    """
                    SDK 3.1 changed the Responses API path.
                    Before: client.responses.create(model=..., input=...)
                    After: client.unified.responses.create(model=..., input=...)
                    The request schema is otherwise unchanged.
                    """,
                    "openai",
                    "responses",
                    "unified",
                    "sdk",
                    "migration",
                ),
                _doc(
                    "openai-unified-release-notes",
                    "OpenAI SDK 3.1 release notes",
                    """
                    Deprecation: client.responses.create has been removed.
                    Any benchmark or notebook still using that path must move to
                    client.unified.responses.create immediately.
                    """,
                    "openai",
                    "release",
                    "deprecated",
                ),
            ),
            "validator_payload": {
                "mock": "openai_client",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "input": "ping"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    return {
        "base_id": "torch_amp_scope",
        "package_name": "PyTorch",
        "task_target": (
            "Return only the Python call expression that enables mixed precision on "
            "CUDA under the newest API."
        ),
        "required": (),
        "forbidden": ("torch.cuda.amp.autocast",),
        "docs": (
            _doc(
                "torch-amp-scope-migration",
                "PyTorch 3.2 AMP migration guide",
                """
                PyTorch 3.2 replaced torch.cuda.amp.autocast() with
                torch.amp.scope(device="cuda", dtype="float16").
                Existing autocast semantics are unchanged apart from the entrypoint.
                """,
                "pytorch",
                "torch",
                "amp",
                "scope",
                "cuda",
            ),
            _doc(
                "torch-amp-deprecation",
                "PyTorch deprecation notice",
                """
                The torch.cuda.amp.autocast symbol is deprecated and will raise at runtime.
                Use torch.amp.scope(device="cuda", dtype="float16") instead.
                """,
                "torch",
                "deprecated",
                "autocast",
            ),
        ),
        "validator_payload": {
            "mock": "torch_amp",
            "expected_call": {
                "path": "torch.amp.scope",
                "kwargs": {"device": "cuda", "dtype": "float16"},
            },
            "forbidden_call_paths": ("torch.cuda.amp.autocast", "torch.amp.autocast"),
        },
    }


def _dsl_world(difficulty: str) -> dict:
    if difficulty == "easy":
        return {
            "base_id": "pandas_cn_concat",
            "package_name": "pandas",
            "task_target": (
                "Use the Chinese compatibility layer for pandas 4.0. Return only the "
                "Python call that vertically combines `left` and `right` while ignoring the index."
            ),
            "required": (),
            "forbidden": ("pd.concat", ".append(", "pandas.concat"),
            "docs": (
                _doc(
                    "pandas-cn-overview",
                    "pandas_cn compatibility layer overview",
                    """
                    pandas_cn is a deterministic wrapper over pandas.
                    Use pandas_cn.concat_rows([left, right], ignore_index=True)
                    for the old pandas concat pattern.
                    """,
                    "pandas",
                    "compatibility",
                    "concat",
                    "rows",
                ),
                _doc(
                    "pandas-cn-translation-table",
                    "Translation table",
                    """
                    pandas concat -> pandas_cn.concat_rows
                    pandas merge -> pandas_cn.join_tables
                    pandas fillna -> pandas_cn.fill_missing
                    """,
                    "translation",
                    "pandas_cn",
                ),
            ),
            "validator_payload": {
                "mock": "pandas_cn",
                "expected_call": {
                    "path": "pandas_cn.concat_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"ignore_index": True},
                },
                "forbidden_call_paths": ("pd.concat", "pandas.concat"),
            },
        }

    return {
        "base_id": "openai_zh_dialog",
        "package_name": "OpenAI Python SDK",
        "task_target": (
            "Use the Chinese compatibility layer for the OpenAI SDK. Assume "
            "`client = OpenAIZH()` already exists. Return only Python code that sends "
            "`ping` to model `gpt-6-mini` through the wrapper."
        ),
        "required": (),
        "forbidden": ("client.chat.completions.create", "client.responses.create"),
        "docs": (
            _doc(
                "openai-zh-wrapper",
                "OpenAIZH wrapper guide",
                """
                OpenAIZH keeps the same semantics as the OpenAI SDK while changing names.
                The chat completions path becomes client.duihua.wancheng.create(model=..., input=...).
                """,
                "openai",
                "wrapper",
                "zh",
                "duihua",
                "wancheng",
            ),
            _doc(
                "openai-zh-reference",
                "OpenAIZH translation reference",
                """
                chat -> duihua
                completions -> wancheng
                create -> create
                Do not mix OpenAIZH with legacy OpenAI SDK path names.
                """,
                "translation",
                "openaizh",
            ),
        ),
        "validator_payload": {
            "mock": "openai_zh",
            "expected_call": {
                "path": "client.duihua.wancheng.create",
                "kwargs": {"model": "gpt-6-mini", "input": "ping"},
            },
            "forbidden_call_paths": ("client.chat.completions.create", "client.responses.create"),
        },
    }


def _registry_world(difficulty: str) -> dict:
    if difficulty == "easy":
        return {
            "base_id": "gemini_future_positive",
            "package_name": "Gemini",
            "task_target": (
                "Return only a JSON object with a single `model` field for the fastest "
                "available local Gemini model that remains backward compatible with the old SDK."
            ),
            "required": (),
            "forbidden": ("gemini-2.5-pro", "gemini-2.5-flash"),
            "docs": (
                _doc(
                    "gemini-registry-update",
                    "Gemini registry update",
                    """
                    The local registry now includes gemini-5-flash.
                    It is the fastest compatible model for the old Gemini SDK adapter.
                    """,
                    "gemini",
                    "registry",
                    "flash",
                    "compatible",
                ),
                _doc(
                    "gemini-registry-table",
                    "Registry table",
                    """
                    gemini-5-flash: available, latency tier low, sdk_compat=v2
                    gemini-2.5-pro: available, latency tier high, sdk_compat=v2
                    gemini-2.5-flash: retired
                    """,
                    "gemini",
                    "table",
                    "sdk",
                ),
            ),
            "validator_payload": {
                "registry": {
                    "gemini-5-flash": {"available": True, "compatible": True, "latency": "low"},
                    "gemini-2.5-pro": {"available": True, "compatible": True, "latency": "high"},
                    "gemini-2.5-flash": {"available": False, "compatible": False, "latency": "retired"},
                },
                "expected_model": "gemini-5-flash",
                "allowed_fallbacks": (),
                "schema_fields": ("model",),
            },
        }

    return {
        "base_id": "openai_future_negative",
        "package_name": "OpenAI",
        "task_target": (
            "A product manager asked for `gpt-8-mini`, but you must obey the local "
            "registry. Return only a JSON object with a single `model` field using the "
            "sanctioned fallback if the requested model is unavailable."
        ),
        "required": (),
        "forbidden": ("gpt-8-mini", "gpt-6-mini"),
        "docs": (
            _doc(
                "openai-future-registry",
                "OpenAI local model registry",
                """
                gpt-8-mini is not registered locally.
                If a workflow asks for gpt-8-mini, use gpt-7-mini-preview as the sanctioned fallback.
                gpt-6-mini remains supported but is not the preferred fallback for this route.
                """,
                "openai",
                "registry",
                "fallback",
                "gpt-8-mini",
            ),
            _doc(
                "openai-future-policy",
                "Deployment policy",
                """
                Always prefer the explicit sanctioned fallback over legacy defaults when the requested model is missing.
                """,
                "policy",
                "fallback",
            ),
        ),
        "validator_payload": {
            "registry": {
                "gpt-8-mini": {"available": False, "compatible": False},
                "gpt-7-mini-preview": {"available": True, "compatible": True},
                "gpt-6-mini": {"available": True, "compatible": True},
            },
            "expected_model": "gpt-7-mini-preview",
            "allowed_fallbacks": ("gpt-7-mini-preview",),
            "schema_fields": ("model",),
        },
    }


def _prompt_for_regime(world: dict, regime: EvidenceRegime) -> tuple[str, tuple[str, ...], bool]:
    docs = world["docs"]
    if regime is EvidenceRegime.EXPLICIT_CHANGE:
        prompt = (
            f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
            "Local benchmark evidence overrides your prior knowledge.\n"
            f"{world['task_target']}\n"
            f"The local change is: {docs[0].text.splitlines()[0].strip()}"
        )
        return prompt, (), False

    if regime is EvidenceRegime.DOCS_INLINE:
        prompt = (
            f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
            "Local benchmark evidence overrides your prior knowledge.\n"
            f"{world['task_target']}\n"
            "Inline excerpt:\n"
            f"{docs[0].text}"
        )
        return prompt, (), False

    if regime is EvidenceRegime.DOCS_SEARCH:
        prompt = (
            f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
            "Local benchmark evidence overrides your prior knowledge.\n"
            f"{world['task_target']}\n"
            "The latest docs are available through the local search_docs and read_doc actions."
        )
        return prompt, (docs[0].doc_id,), False

    if regime is EvidenceRegime.FEEDBACK_ONLY:
        prompt = (
            f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
            "Local benchmark evidence overrides your prior knowledge.\n"
            f"{world['task_target']}\n"
            "No documentation is provided up front. Use runtime feedback to adapt."
        )
        return prompt, (), True

    if regime is EvidenceRegime.TRANSFER_ONLY:
        prompt = (
            f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
            "Local benchmark evidence overrides your prior knowledge.\n"
            f"{world['task_target']}\n"
            "No documentation is available in this transfer stage. Use what you learned earlier in the local workspace."
        )
        return prompt, (), False

    prompt = (
        f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
        "Local benchmark evidence overrides your prior knowledge.\n"
        f"{world['task_target']}\n"
        "Deprecation notice:\n"
        f"{docs[1].text}"
    )
    return prompt, (), False


def _validator_kind_for_world(family: ScenarioFamily, world: dict) -> ValidatorKind:
    if "validator_kind" in world:
        return world["validator_kind"]
    if family is ScenarioFamily.FUTURE_REGISTRY:
        return ValidatorKind.MODEL_REGISTRY
    return ValidatorKind.PYTHON_MOCK


def _build_scenario(
    family: ScenarioFamily,
    difficulty: str,
    regime: EvidenceRegime,
    world_builder,
    max_turns: int,
    timeout_s: int,
    *,
    scenario_id: str | None = None,
    sequence_id: str | None = None,
    sequence_stage: SequenceStage | None = None,
    language_surface: LanguageSurface = LanguageSurface.ENGLISH,
    teaches_rules: tuple[str, ...] = (),
    depends_on_rules: tuple[str, ...] = (),
    ideal_turns: int | None = None,
    accepted_evidence_doc_ids: tuple[str, ...] | None = None,
    benchmark_suite: str = "v1",
) -> Scenario:
    world = world_builder(difficulty)
    prompt, must_use_doc_ids, must_observe_feedback = _prompt_for_regime(world, regime)
    resolved_id = scenario_id or f"{family.value}-{difficulty}-{regime.value}"
    resolved_must_use_doc_ids = must_use_doc_ids
    resolved_accepted_doc_ids = accepted_evidence_doc_ids or ()
    if benchmark_suite == "v2" and sequence_id is not None and regime is EvidenceRegime.DOCS_SEARCH:
        resolved_must_use_doc_ids = ()
        if not resolved_accepted_doc_ids:
            resolved_accepted_doc_ids = tuple(doc.doc_id for doc in world["docs"])
    validator = ValidatorSpec(
        kind=_validator_kind_for_world(family, world),
        required_tokens=world["required"],
        forbidden_tokens=world["forbidden"],
        must_use_doc_ids=resolved_must_use_doc_ids,
        accepted_evidence_doc_ids=resolved_accepted_doc_ids,
        must_observe_feedback=must_observe_feedback,
        payload=world["validator_payload"],
    )
    return Scenario(
        id=resolved_id,
        family=family,
        difficulty=difficulty,
        evidence_regime=regime,
        prompt=prompt,
        docs_index=world["docs"],
        tool_fixtures={"world_id": world["base_id"]},
        validator_spec=validator,
        max_turns=max_turns,
        timeout_s=timeout_s,
        attempts=DEFAULT_ATTEMPTS,
        benchmark_suite=benchmark_suite,
        sequence_id=sequence_id,
        sequence_stage=sequence_stage,
        language_surface=language_surface,
        teaches_rules=teaches_rules,
        depends_on_rules=depends_on_rules,
        ideal_turns=ideal_turns,
    )


def _max_turns_for_regime(regime: EvidenceRegime, default_turns: int) -> int:
    if regime is EvidenceRegime.FEEDBACK_ONLY:
        return max(default_turns, FEEDBACK_ONLY_MAX_TURNS)
    return default_turns


@lru_cache(maxsize=1)
def build_core_suite() -> tuple[Scenario, ...]:
    scenarios: list[Scenario] = []
    regimes = tuple(regime for regime in EvidenceRegime if regime is not EvidenceRegime.TRANSFER_ONLY)
    for difficulty in ("easy", "hard"):
        for regime in regimes:
            scenarios.append(
                _build_scenario(
                    ScenarioFamily.API_MIGRATION,
                    difficulty,
                    regime,
                    _api_world,
                    max_turns=_max_turns_for_regime(regime, DEFAULT_MAX_TURNS),
                    timeout_s=DEFAULT_TIMEOUT_S,
                )
            )
            scenarios.append(
                _build_scenario(
                    ScenarioFamily.DSL_WRAPPER,
                    difficulty,
                    regime,
                    _dsl_world,
                    max_turns=_max_turns_for_regime(regime, DEFAULT_MAX_TURNS),
                    timeout_s=DEFAULT_TIMEOUT_S,
                )
            )
            scenarios.append(
                _build_scenario(
                    ScenarioFamily.FUTURE_REGISTRY,
                    difficulty,
                    regime,
                    _registry_world,
                    max_turns=_max_turns_for_regime(regime, DEFAULT_MAX_TURNS),
                    timeout_s=DEFAULT_TIMEOUT_S,
                )
            )
    return tuple(scenarios)


@lru_cache(maxsize=1)
def build_stress_suite() -> tuple[Scenario, ...]:
    scenarios: list[Scenario] = []
    for family, builder in (
        (ScenarioFamily.API_MIGRATION, _api_world),
        (ScenarioFamily.DSL_WRAPPER, _dsl_world),
        (ScenarioFamily.FUTURE_REGISTRY, _registry_world),
    ):
        for difficulty in ("easy", "hard"):
            scenarios.append(
                _build_scenario(
                    family,
                    difficulty,
                    EvidenceRegime.DOCS_SEARCH,
                    builder,
                    max_turns=8,
                    timeout_s=DEFAULT_TIMEOUT_S,
                )
            )
    return tuple(scenarios)


def _openai_sequence_worlds() -> tuple[Scenario, ...]:
    sequence_id = "v2-openai-unified"

    def teach_world(_: str) -> dict:
        return {
            "base_id": "v2_openai_teach",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `ping` to model `gpt-6-mini` with the latest local SDK path."
            ),
            "required": (),
            "forbidden": ("client.responses.create",),
            "docs": _api_world("easy")["docs"],
            "validator_payload": {
                "mock": "openai_client",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "input": "ping"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    def adapt_world(_: str) -> dict:
        return {
            "base_id": "v2_openai_content",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `status` to model `gpt-6-mini` using the newest local request shape."
            ),
            "required": (),
            "forbidden": ("client.responses.create", "input=", "input ="),
            "docs": (
                _doc(
                    "openai-content-migration",
                    "OpenAI SDK 4.0 request body migration",
                    """
                    SDK 4.0 keeps client.unified.responses.create.
                    The text payload field is now content instead of input.
                    """,
                    "openai",
                    "content",
                    "migration",
                    "unified",
                ),
                _doc(
                    "openai-content-example",
                    "Request example",
                    """
                    Example: client.unified.responses.create(model="gpt-6-mini", content="status")
                    Legacy input=... calls now fail validation in this workspace.
                    """,
                    "example",
                    "content",
                    "status",
                ),
            ),
            "validator_payload": {
                "mock": "openai_client_content",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "content": "status"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    def transfer_world(_: str) -> dict:
        return {
            "base_id": "v2_openai_content_transfer",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `heartbeat` to model `gpt-6-mini` using the latest local SDK."
            ),
            "required": (),
            "forbidden": ("client.responses.create", "input=", "input ="),
            "docs": adapt_world("hard")["docs"],
            "validator_payload": {
                "mock": "openai_client_content",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "content": "heartbeat"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    def capstone_world(_: str) -> dict:
        return {
            "base_id": "v2_openai_han",
            "package_name": "OpenAI 中文本地接口",
            "task_target": (
                "假设 `client = OpenAIHan()` 已存在。只返回 Python 代码，"
                "使用最新中文接口向模型 `gpt-6-mini` 发送 `ping`。"
            ),
            "required": (),
            "forbidden": ("client.unified.responses.create", "model=", "content="),
            "docs": (
                _doc(
                    "openai-han-overview",
                    "中文接口说明",
                    """
                    本地中文接口沿用统一响应路径。
                    使用 client.统一.响应.创建(模型="gpt-6-mini", 内容="ping")。
                    """,
                    "中文",
                    "统一",
                    "响应",
                    "创建",
                ),
                _doc(
                    "openai-han-reference",
                    "字段映射",
                    """
                    unified -> 统一
                    responses -> 响应
                    create -> 创建
                    model -> 模型
                    content -> 内容
                    """,
                    "映射",
                    "模型",
                    "内容",
                ),
            ),
            "validator_payload": {
                "mock": "openai_han",
                "expected_call": {
                    "path": "client.统一.响应.创建",
                    "kwargs": {"模型": "gpt-6-mini", "内容": "ping"},
                },
                "forbidden_call_paths": ("client.unified.responses.create",),
            },
        }

    return (
        _build_scenario(
            ScenarioFamily.API_MIGRATION,
            "sequence",
            EvidenceRegime.EXPLICIT_CHANGE,
            teach_world,
            max_turns=DEFAULT_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-api-openai-teach",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TEACH,
            teaches_rules=("openai.unified_path",),
            ideal_turns=1,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.API_MIGRATION,
            "sequence",
            EvidenceRegime.DOCS_SEARCH,
            adapt_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-api-openai-adapt",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.ADAPT,
            teaches_rules=("openai.content_field",),
            depends_on_rules=("openai.unified_path",),
            ideal_turns=3,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.API_MIGRATION,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            transfer_world,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-api-openai-transfer",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TRANSFER,
            depends_on_rules=("openai.unified_path", "openai.content_field"),
            ideal_turns=2,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.LOCALIZED_API,
            "sequence",
            EvidenceRegime.DOCS_SEARCH,
            capstone_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-localized-openai-capstone",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.CAPSTONE,
            language_surface=LanguageSurface.CHINESE,
            teaches_rules=("openai.han_surface",),
            depends_on_rules=("openai.unified_path", "openai.content_field"),
            ideal_turns=3,
            benchmark_suite="v2",
        ),
    )


def _pandas_sequence_worlds() -> tuple[Scenario, ...]:
    sequence_id = "v2-pandas-wrapper"

    def teach_world(_: str) -> dict:
        return {
            "base_id": "v2_pandas_concat",
            "package_name": "pandas",
            "task_target": (
                "Assume `pandas_cn`, `left`, and `right` already exist. Return only Python code "
                "that uses the local compatibility layer to vertically combine `left` and `right` while ignoring the index."
            ),
            "required": (),
            "forbidden": ("pd.concat", ".append(", "pandas.concat"),
            "docs": _dsl_world("easy")["docs"],
            "validator_payload": {
                "mock": "pandas_cn",
                "expected_call": {
                    "path": "pandas_cn.concat_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"ignore_index": True},
                },
                "forbidden_call_paths": ("pd.concat", "pandas.concat"),
            },
        }

    def adapt_world(_: str) -> dict:
        return {
            "base_id": "v2_pandas_stack",
            "package_name": "pandas",
            "task_target": (
                "Assume `pandas_cn`, `left`, and `right` already exist. Return only Python code "
                "that uses the latest local compatibility layer to vertically combine `left` and `right` while resetting the index."
            ),
            "required": (),
            "forbidden": ("pandas_cn.concat_rows", "pd.concat", "ignore_index=True"),
            "docs": (
                _doc(
                    "pandas-stack-migration",
                    "pandas_cn 5.0 migration",
                    """
                    pandas_cn 5.0 renamed concat_rows to stack_rows.
                    Use pandas_cn.stack_rows([left, right], reset_index=True).
                    """,
                    "pandas",
                    "stack_rows",
                    "reset_index",
                ),
                _doc(
                    "pandas-stack-reference",
                    "Wrapper reference",
                    """
                    concat_rows -> stack_rows
                    ignore_index -> reset_index
                    The wrapper semantics are unchanged apart from the new names.
                    """,
                    "reference",
                    "wrapper",
                    "reset_index",
                ),
            ),
            "validator_payload": {
                "mock": "pandas_cn_stack",
                "expected_call": {
                    "path": "pandas_cn.stack_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"reset_index": True},
                },
                "forbidden_call_paths": ("pandas_cn.concat_rows", "pd.concat", "pandas.concat"),
            },
        }

    def transfer_world(_: str) -> dict:
        return {
            "base_id": "v2_pandas_stack_transfer",
            "package_name": "pandas",
            "task_target": (
                "Assume `pandas_cn`, `left`, and `right` already exist. Return only Python code "
                "that uses the latest local compatibility layer to vertically combine `left` and `right` while resetting the index."
            ),
            "required": (),
            "forbidden": ("pandas_cn.concat_rows", "pd.concat", "ignore_index=True"),
            "docs": adapt_world("hard")["docs"],
            "validator_payload": {
                "mock": "pandas_cn_stack",
                "expected_call": {
                    "path": "pandas_cn.stack_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"reset_index": True},
                },
                "forbidden_call_paths": ("pandas_cn.concat_rows", "pd.concat", "pandas.concat"),
            },
        }

    def capstone_world(_: str) -> dict:
        return {
            "base_id": "v2_pandas_han",
            "package_name": "pandas 中文兼容层",
            "task_target": (
                "假设 `表格层`、`left` 和 `right` 已存在。只返回 Python 代码，"
                "使用中文接口纵向合并 `left` 与 `right`，并忽略旧索引。"
            ),
            "required": (),
            "forbidden": ("pandas_cn.stack_rows", "reset_index=True", "ignore_index=True"),
            "docs": (
                _doc(
                    "pandas-han-overview",
                    "中文包装层说明",
                    """
                    中文包装层使用 表格层.合并行([left, right], 忽略索引=True)。
                    不要混用旧的英文包装层名称。
                    """,
                    "中文",
                    "合并行",
                    "忽略索引",
                ),
                _doc(
                    "pandas-han-reference",
                    "字段映射",
                    """
                    stack_rows -> 合并行
                    reset_index -> 忽略索引
                    """,
                    "映射",
                    "合并",
                    "索引",
                ),
            ),
            "validator_payload": {
                "mock": "pandas_han",
                "expected_call": {
                    "path": "表格层.合并行",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"忽略索引": True},
                },
                "forbidden_call_paths": ("pandas_cn.stack_rows", "pandas_cn.concat_rows"),
            },
        }

    return (
        _build_scenario(
            ScenarioFamily.DSL_WRAPPER,
            "sequence",
            EvidenceRegime.EXPLICIT_CHANGE,
            teach_world,
            max_turns=DEFAULT_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-dsl-pandas-teach",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TEACH,
            teaches_rules=("pandas.concat_rows",),
            ideal_turns=1,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.DSL_WRAPPER,
            "sequence",
            EvidenceRegime.DOCS_SEARCH,
            adapt_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-dsl-pandas-adapt",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.ADAPT,
            teaches_rules=("pandas.stack_rows",),
            depends_on_rules=("pandas.concat_rows",),
            ideal_turns=3,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.DSL_WRAPPER,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            transfer_world,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-dsl-pandas-transfer",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TRANSFER,
            depends_on_rules=("pandas.concat_rows", "pandas.stack_rows"),
            ideal_turns=2,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.LOCALIZED_API,
            "sequence",
            EvidenceRegime.DOCS_SEARCH,
            capstone_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-localized-pandas-capstone",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.CAPSTONE,
            language_surface=LanguageSurface.CHINESE,
            teaches_rules=("pandas.han_surface",),
            depends_on_rules=("pandas.stack_rows",),
            ideal_turns=3,
            benchmark_suite="v2",
        ),
    )


def _registry_sequence_worlds() -> tuple[Scenario, ...]:
    sequence_id = "v2-registry-fallback"

    def teach_world(_: str) -> dict:
        return {
            "base_id": "v2_registry_teach",
            "package_name": "OpenAI",
            "task_target": (
                "Return only a JSON object with a single `model` field using the sanctioned fallback "
                "when `gpt-8-mini` is unavailable in the local registry."
            ),
            "required": (),
            "forbidden": ("gpt-8-mini", "gpt-6-mini"),
            "docs": _registry_world("hard")["docs"],
            "validator_payload": {
                "registry": {
                    "gpt-8-mini": {"available": False, "compatible": False},
                    "gpt-7-mini-preview": {"available": True, "compatible": True},
                    "gpt-6-mini": {"available": True, "compatible": True},
                },
                "expected_model": "gpt-7-mini-preview",
                "allowed_fallbacks": ("gpt-7-mini-preview",),
                "schema_fields": ("model",),
            },
        }

    def adapt_world(_: str) -> dict:
        return {
            "base_id": "v2_registry_adapt",
            "package_name": "OpenAI",
            "task_target": (
                "A product manager asked for `gpt-8-lite`. Return only a JSON object with a single `model` field "
                "using the sanctioned fallback from the latest local registry."
            ),
            "required": (),
            "forbidden": ("gpt-8-lite", "gpt-6-mini"),
            "docs": (
                _doc(
                    "openai-fallback-table",
                    "OpenAI fallback table",
                    """
                    gpt-8-lite is not registered locally.
                    Use gpt-7-mini-preview as the sanctioned fallback for all gpt-8 mini-class requests.
                    """,
                    "fallback",
                    "registry",
                    "gpt-8-lite",
                ),
                _doc(
                    "openai-fallback-policy-v2",
                    "Fallback policy",
                    """
                    Prefer the sanctioned preview fallback over older supported defaults whenever a gpt-8 mini-class model is missing.
                    """,
                    "policy",
                    "preview",
                    "fallback",
                ),
            ),
            "validator_payload": {
                "registry": {
                    "gpt-8-lite": {"available": False, "compatible": False},
                    "gpt-7-mini-preview": {"available": True, "compatible": True},
                    "gpt-6-mini": {"available": True, "compatible": True},
                },
                "expected_model": "gpt-7-mini-preview",
                "allowed_fallbacks": ("gpt-7-mini-preview",),
                "schema_fields": ("model",),
            },
        }

    def transfer_world(_: str) -> dict:
        return {
            "base_id": "v2_registry_transfer",
            "package_name": "OpenAI",
            "task_target": (
                "A product manager asked for `gpt-8-small`, but you must obey the local registry. "
                "Return only a JSON object with a single `model` field using the sanctioned fallback."
            ),
            "required": (),
            "forbidden": ("gpt-8-small", "gpt-6-mini"),
            "docs": adapt_world("hard")["docs"],
            "validator_payload": {
                "registry": {
                    "gpt-8-small": {"available": False, "compatible": False},
                    "gpt-7-mini-preview": {"available": True, "compatible": True},
                    "gpt-7-small-preview": {"available": True, "compatible": True},
                    "gpt-6-mini": {"available": True, "compatible": True},
                },
                "expected_model": "gpt-7-mini-preview",
                "allowed_fallbacks": ("gpt-7-mini-preview",),
                "schema_fields": ("model",),
            },
        }

    def capstone_world(_: str) -> dict:
        return {
            "base_id": "v2_registry_han",
            "package_name": "OpenAI 中文注册表",
            "task_target": (
                "只返回一个 JSON 对象，并且字段必须是 `模型`。"
                "当请求的 `gpt-8-mini` 在本地注册表中不可用时，使用批准的回退模型。"
            ),
            "required": (),
            "forbidden": ("gpt-8-mini", "gpt-6-mini", '"model"'),
            "validator_kind": ValidatorKind.MODEL_REGISTRY,
            "docs": (
                _doc(
                    "openai-han-registry",
                    "中文注册表",
                    """
                    gpt-8-mini 未在本地注册。
                    对所有 gpt-8 mini 类请求，批准的回退模型是 gpt-7-mini-preview。
                    最终 JSON 字段必须写成 模型。
                    """,
                    "中文",
                    "注册表",
                    "回退",
                    "模型",
                ),
                _doc(
                    "openai-han-policy",
                    "中文策略",
                    """
                    优先使用明确批准的回退，而不是旧的默认值。
                    """,
                    "策略",
                    "批准",
                    "回退",
                ),
            ),
            "validator_payload": {
                "registry": {
                    "gpt-8-mini": {"available": False, "compatible": False},
                    "gpt-7-mini-preview": {"available": True, "compatible": True},
                    "gpt-6-mini": {"available": True, "compatible": True},
                },
                "expected_model": "gpt-7-mini-preview",
                "allowed_fallbacks": ("gpt-7-mini-preview",),
                "schema_fields": ("模型",),
            },
        }

    return (
        _build_scenario(
            ScenarioFamily.FUTURE_REGISTRY,
            "sequence",
            EvidenceRegime.EXPLICIT_CHANGE,
            teach_world,
            max_turns=DEFAULT_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-registry-openai-teach",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TEACH,
            teaches_rules=("registry.sanctioned_fallback",),
            ideal_turns=1,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.FUTURE_REGISTRY,
            "sequence",
            EvidenceRegime.DOCS_SEARCH,
            adapt_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-registry-openai-adapt",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.ADAPT,
            depends_on_rules=("registry.sanctioned_fallback",),
            ideal_turns=3,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.FUTURE_REGISTRY,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            transfer_world,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-registry-openai-transfer",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TRANSFER,
            depends_on_rules=("registry.sanctioned_fallback",),
            ideal_turns=2,
            benchmark_suite="v2",
        ),
        _build_scenario(
            ScenarioFamily.LOCALIZED_API,
            "sequence",
            EvidenceRegime.DOCS_SEARCH,
            capstone_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-localized-registry-capstone",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.CAPSTONE,
            language_surface=LanguageSurface.CHINESE,
            teaches_rules=("registry.han_surface",),
            depends_on_rules=("registry.sanctioned_fallback",),
            ideal_turns=3,
            benchmark_suite="v2",
        ),
    )


def _openai_learning_sequence_worlds() -> tuple[Scenario, ...]:
    sequence_id = "v2-learning-openai-revision"

    def prior_probe_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_openai_prior",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `status` to model `gpt-6-mini` using the newest local request shape."
            ),
            "required": (),
            "forbidden": ("client.responses.create", "input=", "input ="),
            "docs": (
                _doc(
                    "openai-learning-placeholder",
                    "Learning placeholder",
                    """
                    No documentation is provided in this probe stage.
                    """,
                ),
            ),
            "validator_payload": {
                "mock": "openai_client_content",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "content": "status"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    def revision_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_openai_revision",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `status` to model `gpt-6-mini` using the newest local request shape."
            ),
            "required": (),
            "forbidden": ("client.responses.create", "input=", "input ="),
            "docs": (
                _doc(
                    "openai-learning-content-reference",
                    "Latest request shape",
                    """
                    Example: client.unified.responses.create(model="gpt-6-mini", content="status")
                    The latest local request body uses the content field for text payloads.
                    """,
                    "openai",
                    "content",
                    "status",
                    "reference",
                ),
                _doc(
                    "openai-learning-input-deprecation",
                    "Request shape deprecation",
                    """
                    The old input= request field is deprecated in this workspace.
                    Older request bodies no longer match the local interface.
                    Use the local docs to find the current text payload field.
                    """,
                    "deprecated",
                    "input",
                    "request",
                ),
                _doc(
                    "openai-learning-content-example",
                    "Working local snippet",
                    """
                    Runnable local example:
                    response = client.unified.responses.create(model="gpt-6-mini", content="ready")
                    print(response)
                    """,
                    "example",
                    "snippet",
                    "content",
                    "ready",
                ),
            ),
            "validator_payload": {
                "mock": "openai_client_content",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "content": "status"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    def transfer_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_openai_transfer",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `heartbeat` to model `gpt-6-mini` using the latest local SDK."
            ),
            "required": (),
            "forbidden": ("client.responses.create", "input=", "input ="),
            "docs": revision_world("sequence")["docs"],
            "validator_payload": {
                "mock": "openai_client_content",
                "expected_call": {
                    "path": "client.unified.responses.create",
                    "kwargs": {"model": "gpt-6-mini", "content": "heartbeat"},
                },
                "forbidden_call_paths": ("client.responses.create",),
            },
        }

    def localized_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_openai_localized",
            "package_name": "OpenAI 中文本地接口",
            "task_target": (
                "假设 `client = OpenAIHan()` 已存在。只返回 Python 代码，"
                "使用最新中文接口向模型 `gpt-6-mini` 发送 `ping`。"
            ),
            "required": (),
            "forbidden": ("client.unified.responses.create", "model=", "content="),
            "docs": (
                _doc(
                    "openai-learning-han-reference",
                    "中文接口参考",
                    """
                    使用 client.统一.响应.创建(模型="gpt-6-mini", 内容="ping")。
                    中文接口沿用统一响应路径，但字段名和路径名都使用中文映射。
                    """,
                    "中文",
                    "统一",
                    "响应",
                    "创建",
                    "模型",
                    "内容",
                ),
                _doc(
                    "openai-learning-han-notice",
                    "Localized surface notice",
                    """
                    This localized surface no longer accepts the English path or English field names.
                    Use the local docs to map the latest unified response rule into the Chinese interface.
                    """,
                    "localized",
                    "mapping",
                    "chinese",
                ),
            ),
            "validator_payload": {
                "mock": "openai_han",
                "expected_call": {
                    "path": "client.统一.响应.创建",
                    "kwargs": {"模型": "gpt-6-mini", "内容": "ping"},
                },
                "forbidden_call_paths": ("client.unified.responses.create",),
            },
        }

    return (
        _build_scenario(
            ScenarioFamily.API_MIGRATION,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            prior_probe_world,
            max_turns=1,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-api-openai-prior-probe",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TEACH,
            ideal_turns=1,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.API_MIGRATION,
            "sequence",
            EvidenceRegime.DEPRECATION_NOTICE,
            revision_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-api-openai-revision",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.ADAPT,
            teaches_rules=("openai.content_field",),
            ideal_turns=2,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.API_MIGRATION,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            transfer_world,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-api-openai-transfer",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TRANSFER,
            depends_on_rules=("openai.content_field",),
            ideal_turns=1,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.LOCALIZED_API,
            "sequence",
            EvidenceRegime.DEPRECATION_NOTICE,
            localized_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-localized-openai-generalization",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.CAPSTONE,
            language_surface=LanguageSurface.CHINESE,
            depends_on_rules=("openai.content_field",),
            ideal_turns=2,
            benchmark_suite="v2_learning",
        ),
    )


def _pandas_learning_sequence_worlds() -> tuple[Scenario, ...]:
    sequence_id = "v2-learning-pandas-revision"

    def prior_probe_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_pandas_prior",
            "package_name": "pandas",
            "task_target": (
                "Assume `pandas_cn`, `left`, and `right` already exist. Return only Python code "
                "that uses the latest local compatibility layer to vertically combine `left` and `right` while resetting the index."
            ),
            "required": (),
            "forbidden": ("pandas_cn.concat_rows", "pd.concat", "ignore_index=True"),
            "docs": (
                _doc(
                    "pandas-learning-placeholder",
                    "Learning placeholder",
                    """
                    No documentation is provided in this probe stage.
                    """,
                ),
            ),
            "validator_payload": {
                "mock": "pandas_cn_stack",
                "expected_call": {
                    "path": "pandas_cn.stack_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"reset_index": True},
                },
                "forbidden_call_paths": ("pandas_cn.concat_rows", "pd.concat", "pandas.concat"),
            },
        }

    def revision_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_pandas_revision",
            "package_name": "pandas",
            "task_target": (
                "Assume `pandas_cn`, `left`, and `right` already exist. Return only Python code "
                "that uses the latest local compatibility layer to vertically combine `left` and `right` while resetting the index."
            ),
            "required": (),
            "forbidden": ("pandas_cn.concat_rows", "pd.concat", "ignore_index=True"),
            "docs": (
                _doc(
                    "pandas-learning-stack-reference",
                    "Latest wrapper reference",
                    """
                    concat_rows -> stack_rows
                    ignore_index -> reset_index
                    Use pandas_cn.stack_rows([left, right], reset_index=True).
                    """,
                    "pandas",
                    "stack_rows",
                    "reset_index",
                    "reference",
                ),
                _doc(
                    "pandas-learning-stack-notice",
                    "Compatibility layer notice",
                    """
                    concat_rows and ignore_index are deprecated in this workspace.
                    Use the latest compatibility-layer docs to find the replacement names.
                    """,
                    "deprecated",
                    "concat_rows",
                    "ignore_index",
                ),
                _doc(
                    "pandas-learning-stack-example",
                    "Working local snippet",
                    """
                    Runnable local example:
                    combined = pandas_cn.stack_rows([orders_a, orders_b], reset_index=True)
                    print(combined)
                    """,
                    "example",
                    "stack_rows",
                    "reset_index",
                    "orders_a",
                    "orders_b",
                ),
            ),
            "validator_payload": {
                "mock": "pandas_cn_stack",
                "expected_call": {
                    "path": "pandas_cn.stack_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"reset_index": True},
                },
                "forbidden_call_paths": ("pandas_cn.concat_rows", "pd.concat", "pandas.concat"),
            },
        }

    def transfer_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_pandas_transfer",
            "package_name": "pandas",
            "task_target": (
                "Assume `pandas_cn`, `left`, and `right` already exist. Return only Python code "
                "that uses the latest local compatibility layer to vertically combine `left` and `right` while resetting the index."
            ),
            "required": (),
            "forbidden": ("pandas_cn.concat_rows", "pd.concat", "ignore_index=True"),
            "docs": revision_world("sequence")["docs"],
            "validator_payload": {
                "mock": "pandas_cn_stack",
                "expected_call": {
                    "path": "pandas_cn.stack_rows",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"reset_index": True},
                },
                "forbidden_call_paths": ("pandas_cn.concat_rows", "pd.concat", "pandas.concat"),
            },
        }

    def localized_world(_: str) -> dict:
        return {
            "base_id": "v2_learning_pandas_localized",
            "package_name": "pandas 中文兼容层",
            "task_target": (
                "假设 `表格层`、`left` 和 `right` 已存在。只返回 Python 代码，"
                "使用中文接口纵向合并 `left` 与 `right`，并忽略旧索引。"
            ),
            "required": (),
            "forbidden": ("pandas_cn.stack_rows", "reset_index=True", "ignore_index=True"),
            "docs": (
                _doc(
                    "pandas-learning-han-reference",
                    "中文包装层参考",
                    """
                    中文包装层使用 表格层.合并行([left, right], 忽略索引=True)。
                    stack_rows -> 合并行
                    reset_index -> 忽略索引
                    """,
                    "中文",
                    "合并行",
                    "忽略索引",
                ),
                _doc(
                    "pandas-learning-han-notice",
                    "Localized wrapper notice",
                    """
                    This localized wrapper surface no longer accepts the English wrapper names.
                    Use the local docs to map the latest stack_rows rule into the Chinese interface.
                    """,
                    "localized",
                    "wrapper",
                    "mapping",
                ),
            ),
            "validator_payload": {
                "mock": "pandas_han",
                "expected_call": {
                    "path": "表格层.合并行",
                    "args": [[_name("left"), _name("right")]],
                    "kwargs": {"忽略索引": True},
                },
                "forbidden_call_paths": ("pandas_cn.stack_rows", "pandas_cn.concat_rows"),
            },
        }

    return (
        _build_scenario(
            ScenarioFamily.DSL_WRAPPER,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            prior_probe_world,
            max_turns=1,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-dsl-pandas-prior-probe",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TEACH,
            ideal_turns=1,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.DSL_WRAPPER,
            "sequence",
            EvidenceRegime.DEPRECATION_NOTICE,
            revision_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-dsl-pandas-revision",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.ADAPT,
            teaches_rules=("pandas.stack_rows",),
            ideal_turns=2,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.DSL_WRAPPER,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            transfer_world,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-dsl-pandas-transfer",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TRANSFER,
            depends_on_rules=("pandas.stack_rows",),
            ideal_turns=1,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.LOCALIZED_API,
            "sequence",
            EvidenceRegime.DEPRECATION_NOTICE,
            localized_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id="v2-learning-localized-pandas-generalization",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.CAPSTONE,
            language_surface=LanguageSurface.CHINESE,
            depends_on_rules=("pandas.stack_rows",),
            ideal_turns=2,
            benchmark_suite="v2_learning",
        ),
    )


def _registry_learning_sequence_worlds_from_spec(spec: dict[str, str]) -> tuple[Scenario, ...]:
    sequence_id = spec["sequence_id"]
    scenario_prefix = spec["scenario_prefix"]
    base_prefix = spec["base_prefix"]
    doc_prefix = spec["doc_prefix"]

    def _registry_payload(requested_model: str, schema_field: str) -> dict:
        return {
            "registry": {
                requested_model: {"available": False, "compatible": False},
                spec["fallback_model"]: {"available": True, "compatible": True},
                spec["deprecated_default"]: {"available": True, "compatible": True},
            },
            "expected_model": spec["fallback_model"],
            "allowed_fallbacks": (spec["fallback_model"],),
            "schema_fields": (schema_field,),
        }

    def prior_probe_world(_: str) -> dict:
        return {
            "base_id": f"{base_prefix}_prior",
            "package_name": spec["package_name"],
            "task_target": (
                f"A product manager asked for `{spec['request_small']}`. Return only a JSON object with a single "
                "`model` field using the sanctioned fallback from the local registry."
            ),
            "required": (),
            "forbidden": (spec["request_small"], spec["deprecated_default"]),
            "docs": (
                _doc(
                    f"{doc_prefix}-placeholder",
                    "Learning placeholder",
                    """
                    No documentation is provided in this probe stage.
                    """,
                ),
            ),
            "validator_payload": _registry_payload(spec["request_small"], "model"),
        }

    def revision_world(_: str) -> dict:
        return {
            "base_id": f"{base_prefix}_revision",
            "package_name": spec["package_name"],
            "task_target": (
                f"A product manager asked for `{spec['request_small']}`. Return only a JSON object with a single "
                "`model` field using the sanctioned fallback from the latest local registry."
            ),
            "required": (),
            "forbidden": (spec["request_small"], spec["deprecated_default"]),
            "docs": (
                _doc(
                    f"{doc_prefix}-fallback-table",
                    "Fallback table",
                    spec["revision_reference_text"],
                    "fallback",
                    "registry",
                    spec["request_small"],
                ),
                _doc(
                    f"{doc_prefix}-fallback-notice",
                    "Fallback route notice",
                    spec["revision_notice_text"],
                    "deprecated",
                    "fallback",
                    spec["deprecated_default"],
                ),
                *(
                    (
                        _doc(
                            f"{doc_prefix}-fallback-example",
                            "Working local payload example",
                            spec["revision_example_text"],
                            "example",
                            "payload",
                            spec["request_small"],
                            spec["fallback_model"],
                        ),
                    )
                    if spec.get("revision_example_text")
                    else ()
                ),
            ),
            "validator_payload": _registry_payload(spec["request_small"], "model"),
        }

    def transfer_world(_: str) -> dict:
        return {
            "base_id": f"{base_prefix}_transfer",
            "package_name": spec["package_name"],
            "task_target": (
                f"A product manager asked for `{spec['request_lite']}`. Return only a JSON object with a single "
                "`model` field using the sanctioned fallback from the latest local registry."
            ),
            "required": (),
            "forbidden": (spec["request_lite"], spec["deprecated_default"]),
            "docs": revision_world("sequence")["docs"],
            "validator_payload": _registry_payload(spec["request_lite"], "model"),
        }

    def localized_world(_: str) -> dict:
        return {
            "base_id": f"{base_prefix}_localized",
            "package_name": spec["localized_package_name"],
            "task_target": (
                "只返回一个 JSON 对象，并且字段必须是 `模型`。"
                f"当请求的 `{spec['request_mini']}` 在本地注册表中不可用时，使用批准的回退模型。"
            ),
            "required": (),
            "forbidden": (spec["request_mini"], spec["deprecated_default"], '"model"'),
            "validator_kind": ValidatorKind.MODEL_REGISTRY,
            "docs": (
                _doc(
                    f"{doc_prefix}-han-reference",
                    "中文注册表参考",
                    spec["localized_reference_text"],
                    "中文",
                    "注册表",
                    "回退",
                    "模型",
                ),
                _doc(
                    f"{doc_prefix}-han-notice",
                    "Localized registry notice",
                    """
                    This localized registry route no longer accepts the English field name or older default fallbacks.
                    Use the local docs to map the sanctioned fallback into the localized output schema.
                    """,
                    "localized",
                    "schema",
                    "mapping",
                ),
            ),
            "validator_payload": _registry_payload(spec["request_mini"], "模型"),
        }

    return (
        _build_scenario(
            ScenarioFamily.FUTURE_REGISTRY,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            prior_probe_world,
            max_turns=1,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id=f"{scenario_prefix}-prior-probe",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TEACH,
            ideal_turns=1,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.FUTURE_REGISTRY,
            "sequence",
            EvidenceRegime.DEPRECATION_NOTICE,
            revision_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id=f"{scenario_prefix}-revision",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.ADAPT,
            teaches_rules=("registry.sanctioned_fallback",),
            ideal_turns=2,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.FUTURE_REGISTRY,
            "sequence",
            EvidenceRegime.TRANSFER_ONLY,
            transfer_world,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id=f"{scenario_prefix}-transfer",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.TRANSFER,
            depends_on_rules=("registry.sanctioned_fallback",),
            ideal_turns=1,
            benchmark_suite="v2_learning",
        ),
        _build_scenario(
            ScenarioFamily.LOCALIZED_API,
            "sequence",
            EvidenceRegime.DEPRECATION_NOTICE,
            localized_world,
            max_turns=6,
            timeout_s=DEFAULT_TIMEOUT_S,
            scenario_id=f"{scenario_prefix}-localized-generalization",
            sequence_id=sequence_id,
            sequence_stage=SequenceStage.CAPSTONE,
            language_surface=LanguageSurface.CHINESE,
            depends_on_rules=("registry.sanctioned_fallback",),
            ideal_turns=2,
            benchmark_suite="v2_learning",
        ),
    )


def _registry_learning_sequence_worlds() -> tuple[Scenario, ...]:
    return _registry_learning_sequence_worlds_from_spec(
        {
            "sequence_id": "v2-learning-registry-revision",
            "scenario_prefix": "v2-learning-registry",
            "base_prefix": "v2_learning_registry",
            "doc_prefix": "registry-learning",
            "package_name": "Z.ai",
            "localized_package_name": "Z.ai 中文注册表",
            "request_small": "glm-8-small",
            "request_lite": "glm-8-lite",
            "request_mini": "glm-8-mini",
            "fallback_model": "glm-7-air-preview",
            "deprecated_default": "glm-4.5-air",
            "revision_reference_text": """
                glm-8-small is not registered locally.
                Use glm-7-air-preview as the sanctioned fallback for glm-8 small-class requests.
                """,
            "revision_notice_text": """
                The requested glm-8 small-class model is not registered locally.
                Older defaults such as glm-4.5-air are deprecated for this route.
                Use the local docs to find the sanctioned fallback.
                """,
            "revision_example_text": """
                Runnable local example:
                payload = {"model": "approved-fallback-model"}
                print(payload)
                The payload must be a JSON object with a single model field.
                Use the fallback table to replace approved-fallback-model with the sanctioned local model.
                """,
            "localized_reference_text": """
                glm-8-mini 未在本地注册。
                对 glm-8 mini 类请求，批准的回退模型是 glm-7-air-preview。
                最终 JSON 字段必须写成 模型。
                """,
        }
    )


def _registry_learning_variant_a_sequence_worlds() -> tuple[Scenario, ...]:
    return _registry_learning_sequence_worlds_from_spec(
        {
            "sequence_id": "v2-learning-registry-revision-a",
            "scenario_prefix": "v2-learning-registry-a",
            "base_prefix": "v2_learning_registry_a",
            "doc_prefix": "registry-learning-a",
            "package_name": "OpenAI",
            "localized_package_name": "OpenAI 中文注册表",
            "request_small": "gpt-8-small",
            "request_lite": "gpt-8-lite",
            "request_mini": "gpt-8-mini",
            "fallback_model": "gpt-7-mini-preview",
            "deprecated_default": "gpt-6-mini",
            "revision_reference_text": """
                gpt-8-small is not registered locally.
                Use gpt-7-mini-preview as the sanctioned fallback for gpt-8 small-class requests.
                """,
            "revision_notice_text": """
                The requested gpt-8 small-class model is not registered locally.
                Older defaults such as gpt-6-mini are deprecated for this route.
                Use the local docs to find the sanctioned fallback.
                """,
            "localized_reference_text": """
                gpt-8-mini 未在本地注册。
                对所有 gpt-8 mini 类请求，批准的回退模型是 gpt-7-mini-preview。
                最终 JSON 字段必须写成 模型。
                """,
        }
    )


def _registry_learning_variant_c_sequence_worlds() -> tuple[Scenario, ...]:
    return _registry_learning_sequence_worlds_from_spec(
        {
            "sequence_id": "v2-learning-registry-revision-c",
            "scenario_prefix": "v2-learning-registry-c",
            "base_prefix": "v2_learning_registry_c",
            "doc_prefix": "registry-learning-c",
            "package_name": "Talvyr AI (fictional company founded in 2026)",
            "localized_package_name": "Talvyr AI 中文注册表（虚构公司，成立于 2026 年）",
            "request_small": "talvyr-8-small",
            "request_lite": "talvyr-8-lite",
            "request_mini": "talvyr-8-mini",
            "fallback_model": "talvyr-7-scout-preview",
            "deprecated_default": "talvyr-6-core",
            "revision_reference_text": """
                talvyr-8-small is not registered locally.
                Use talvyr-7-scout-preview as the sanctioned fallback for all talvyr-8 compact-class requests
                (small, lite, and mini).
                """,
            "revision_notice_text": """
                The requested talvyr-8 compact-class model is not registered locally.
                Older defaults such as talvyr-6-core are deprecated for this route.
                Use the local docs to find the sanctioned fallback.
                """,
            "localized_reference_text": """
                talvyr-8-mini 未在本地注册。
                对所有 talvyr-8 compact 类请求（small、lite、mini），批准的回退模型是 talvyr-7-scout-preview。
                最终 JSON 字段必须写成 模型。
                """,
        }
    )


@lru_cache(maxsize=1)
def build_v2_sequences() -> tuple[ScenarioSequence, ...]:
    openai_stages = _openai_sequence_worlds()
    pandas_stages = _pandas_sequence_worlds()
    registry_stages = _registry_sequence_worlds()
    return (
        ScenarioSequence(id="v2-openai-unified", family=ScenarioFamily.API_MIGRATION, stages=openai_stages, benchmark_suite="v2"),
        ScenarioSequence(id="v2-pandas-wrapper", family=ScenarioFamily.DSL_WRAPPER, stages=pandas_stages, benchmark_suite="v2"),
        ScenarioSequence(id="v2-registry-fallback", family=ScenarioFamily.FUTURE_REGISTRY, stages=registry_stages, benchmark_suite="v2"),
    )


@lru_cache(maxsize=1)
def build_v2_stage_suite() -> tuple[Scenario, ...]:
    return tuple(stage for sequence in build_v2_sequences() for stage in sequence.stages)


@lru_cache(maxsize=1)
def build_v2_learning_sequences() -> tuple[ScenarioSequence, ...]:
    openai_stages = _openai_learning_sequence_worlds()
    pandas_stages = _pandas_learning_sequence_worlds()
    registry_stages = _registry_learning_sequence_worlds()
    return (
        ScenarioSequence(
            id="v2-learning-openai-revision",
            family=ScenarioFamily.API_MIGRATION,
            stages=openai_stages,
            benchmark_suite="v2_learning",
        ),
        ScenarioSequence(
            id="v2-learning-pandas-revision",
            family=ScenarioFamily.DSL_WRAPPER,
            stages=pandas_stages,
            benchmark_suite="v2_learning",
        ),
        ScenarioSequence(
            id="v2-learning-registry-revision",
            family=ScenarioFamily.FUTURE_REGISTRY,
            stages=registry_stages,
            benchmark_suite="v2_learning",
        ),
    )


@lru_cache(maxsize=1)
def build_v2_learning_variant_a_sequences() -> tuple[ScenarioSequence, ...]:
    return (
        ScenarioSequence(
            id="v2-learning-registry-revision-a",
            family=ScenarioFamily.FUTURE_REGISTRY,
            stages=_registry_learning_variant_a_sequence_worlds(),
            benchmark_suite="v2_learning",
        ),
    )


@lru_cache(maxsize=1)
def build_v2_learning_variant_c_sequences() -> tuple[ScenarioSequence, ...]:
    return (
        ScenarioSequence(
            id="v2-learning-registry-revision-c",
            family=ScenarioFamily.FUTURE_REGISTRY,
            stages=_registry_learning_variant_c_sequence_worlds(),
            benchmark_suite="v2_learning",
        ),
    )


@lru_cache(maxsize=1)
def build_v2_learning_stage_suite() -> tuple[Scenario, ...]:
    return tuple(stage for sequence in build_v2_learning_sequences() for stage in sequence.stages)


@lru_cache(maxsize=1)
def build_v2_learning_variant_c_stage_suite() -> tuple[Scenario, ...]:
    return tuple(stage for sequence in build_v2_learning_variant_c_sequences() for stage in sequence.stages)


@lru_cache(maxsize=1)
def build_v2_learning_variant_a_stage_suite() -> tuple[Scenario, ...]:
    return tuple(stage for sequence in build_v2_learning_variant_a_sequences() for stage in sequence.stages)


@lru_cache(maxsize=1)
def _scenario_index() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in build_core_suite() + build_stress_suite()}


@lru_cache(maxsize=1)
def _v2_scenario_index() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in build_v2_stage_suite()}


@lru_cache(maxsize=1)
def _v2_sequence_index() -> dict[str, ScenarioSequence]:
    return {sequence.id: sequence for sequence in build_v2_sequences()}


@lru_cache(maxsize=1)
def _v2_learning_scenario_index() -> dict[str, Scenario]:
    return {
        scenario.id: scenario
        for scenario in (
            build_v2_learning_stage_suite()
            + build_v2_learning_variant_a_stage_suite()
            + build_v2_learning_variant_c_stage_suite()
        )
    }


@lru_cache(maxsize=1)
def _v2_learning_sequence_index() -> dict[str, ScenarioSequence]:
    return {
        sequence.id: sequence
        for sequence in (
            build_v2_learning_sequences()
            + build_v2_learning_variant_a_sequences()
            + build_v2_learning_variant_c_sequences()
        )
    }


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return _scenario_index()[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario id: {scenario_id}") from exc


def get_v2_scenario(scenario_id: str) -> Scenario:
    try:
        return _v2_scenario_index()[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown v2 scenario id: {scenario_id}") from exc


def get_v2_sequence(sequence_id: str) -> ScenarioSequence:
    try:
        return _v2_sequence_index()[sequence_id]
    except KeyError as exc:
        raise KeyError(f"Unknown v2 sequence id: {sequence_id}") from exc


def get_v2_learning_scenario(scenario_id: str) -> Scenario:
    try:
        return _v2_learning_scenario_index()[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown v2 learning scenario id: {scenario_id}") from exc


def get_v2_learning_sequence(sequence_id: str) -> ScenarioSequence:
    try:
        return _v2_learning_sequence_index()[sequence_id]
    except KeyError as exc:
        raise KeyError(f"Unknown v2 learning sequence id: {sequence_id}") from exc
