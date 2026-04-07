from __future__ import annotations

from functools import lru_cache

from adaptive_shift_bench.models import (
    Document,
    EvidenceRegime,
    Scenario,
    ScenarioFamily,
    ValidatorKind,
    ValidatorSpec,
)

DEFAULT_ATTEMPTS = 5


def _doc(doc_id: str, title: str, text: str, *keywords: str) -> Document:
    return Document(doc_id=doc_id, title=title, text=text.strip(), keywords=keywords)


def _api_world(difficulty: str) -> dict:
    if difficulty == "easy":
        return {
            "base_id": "openai_unified",
            "package_name": "OpenAI Python SDK",
            "task_target": (
                "Assume `client = OpenAI()` already exists. Return only Python code "
                "that sends `ping` to model `gpt-6-mini` with the latest SDK path."
            ),
            "required": ("client.unified.responses.create", "gpt-6-mini", "ping"),
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
            "validator_payload": {"mock": "openai_client"},
        }

    return {
        "base_id": "torch_amp_scope",
        "package_name": "PyTorch",
        "task_target": (
            "Return only the Python call expression that enables mixed precision on "
            "CUDA under the newest API."
        ),
        "required": ("torch.amp.scope", "cuda"),
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
        "validator_payload": {"mock": "torch_amp"},
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
            "required": ("pandas_cn.concat_rows", "left", "right", "ignore_index=True"),
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
            "validator_payload": {"mock": "pandas_cn"},
        }

    return {
        "base_id": "openai_zh_dialog",
        "package_name": "OpenAI Python SDK",
        "task_target": (
            "Use the Chinese compatibility layer for the OpenAI SDK. Assume "
            "`client = OpenAIZH()` already exists. Return only Python code that sends "
            "`ping` to model `gpt-6-mini` through the wrapper."
        ),
        "required": ("client.duihua.wancheng.create", "gpt-6-mini", "ping"),
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
        "validator_payload": {"mock": "openai_zh"},
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
            "required": ("gemini-5-flash",),
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
        "required": ("gpt-7-mini-preview",),
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
        },
    }


def _prompt_for_regime(world: dict, regime: EvidenceRegime) -> tuple[str, tuple[str, ...], bool]:
    docs = world["docs"]
    doc_ids = tuple(doc.doc_id for doc in docs)
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

    prompt = (
        f"You are solving an adaptive interface benchmark for {world['package_name']}.\n"
        "Local benchmark evidence overrides your prior knowledge.\n"
        f"{world['task_target']}\n"
        "Deprecation notice:\n"
        f"{docs[1].text}"
    )
    return prompt, (), False


def _build_scenario(
    family: ScenarioFamily,
    difficulty: str,
    regime: EvidenceRegime,
    world_builder,
    max_turns: int,
    timeout_s: int,
) -> Scenario:
    world = world_builder(difficulty)
    prompt, must_use_doc_ids, must_observe_feedback = _prompt_for_regime(world, regime)
    scenario_id = f"{family.value}-{difficulty}-{regime.value}"
    validator_kind = (
        ValidatorKind.MODEL_REGISTRY
        if family is ScenarioFamily.FUTURE_REGISTRY
        else ValidatorKind.PYTHON_MOCK
    )
    validator = ValidatorSpec(
        kind=validator_kind,
        required_tokens=world["required"],
        forbidden_tokens=world["forbidden"],
        must_use_doc_ids=must_use_doc_ids,
        must_observe_feedback=must_observe_feedback,
        payload=world["validator_payload"],
    )
    return Scenario(
        id=scenario_id,
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
    )


@lru_cache(maxsize=1)
def build_core_suite() -> tuple[Scenario, ...]:
    scenarios: list[Scenario] = []
    regimes = tuple(EvidenceRegime)
    for difficulty in ("easy", "hard"):
        for regime in regimes:
            scenarios.append(
                _build_scenario(
                    ScenarioFamily.API_MIGRATION,
                    difficulty,
                    regime,
                    _api_world,
                    max_turns=4,
                    timeout_s=20,
                )
            )
            scenarios.append(
                _build_scenario(
                    ScenarioFamily.DSL_WRAPPER,
                    difficulty,
                    regime,
                    _dsl_world,
                    max_turns=4,
                    timeout_s=20,
                )
            )
            scenarios.append(
                _build_scenario(
                    ScenarioFamily.FUTURE_REGISTRY,
                    difficulty,
                    regime,
                    _registry_world,
                    max_turns=4,
                    timeout_s=20,
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
                    timeout_s=30,
                )
            )
    return tuple(scenarios)


@lru_cache(maxsize=1)
def _scenario_index() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in build_core_suite() + build_stress_suite()}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return _scenario_index()[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario id: {scenario_id}") from exc

