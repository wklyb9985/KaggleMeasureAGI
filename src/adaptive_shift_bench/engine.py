from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Literal

from adaptive_shift_bench.llm import LLMAdapter
from adaptive_shift_bench.models import (
    ActionRecord,
    EvidenceRegime,
    EpisodeAction,
    EpisodeResult,
    Scenario,
    ScenarioSequence,
    SequenceResult,
    ScenarioFamily,
    ValidatorKind,
)
from adaptive_shift_bench.parsing import parse_action
from adaptive_shift_bench.reporting import (
    V2_LEARNING_SCORE_WEIGHTS,
    aggregate_attempts,
    aggregate_sequence_results,
)
from adaptive_shift_bench.scenarios import (
    DEFAULT_ATTEMPTS,
    build_core_suite,
    build_stress_suite,
    build_v2_learning_sequences,
    build_v2_sequences,
)


class BenchmarkValidationError(RuntimeError):
    pass


class DeprecatedInterfaceError(RuntimeError):
    pass


class RegistrySelectionError(RuntimeError):
    pass


PromptStyle = Literal["benchmark", "release_note"]

_ALL_ACTION_SPECS = (
    '- {"action":"search_local_docs","query":"..."}',
    '- {"action":"read_local_doc","doc_id":"..."}',
    '- {"action":"run_candidate","candidate":"..."}',
    '- {"action":"select_model","model":"..."}',
    '- {"action":"answer","content":"..."}',
)

_SAFE_BUILTINS: dict[str, Any] = {
    "print": lambda *args, **kwargs: None,
    "len": len,
    "range": range,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "isinstance": isinstance,
    "type": type,
    "Exception": Exception,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "AttributeError": AttributeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
}


@dataclass(frozen=True)
class RecordedCall:
    path: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    raised: bool = False


class _CallLog:
    def __init__(self) -> None:
        self.calls: list[RecordedCall] = []

    def record(
        self,
        path: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        raised: bool = False,
    ) -> None:
        self.calls.append(RecordedCall(path=path, args=args, kwargs=dict(kwargs), raised=raised))


class _RecordingProxy:
    """Generic proxy that wraps a mock object and records all method calls."""

    __slots__ = ("_target", "_path", "_log")

    def __init__(self, target: object, path: str, log: _CallLog) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        path = object.__getattribute__(self, "_path")
        log = object.__getattribute__(self, "_log")
        attr = getattr(target, name)
        child_path = f"{path}.{name}"

        if callable(attr) and not isinstance(attr, type):
            def _recording_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = attr(*args, **kwargs)
                    log.record(child_path, args, kwargs)
                    return result
                except Exception:
                    log.record(child_path, args, kwargs, raised=True)
                    raise
            return _recording_wrapper

        if hasattr(attr, "__dict__") and not isinstance(attr, (str, int, float, bool)):
            return _RecordingProxy(attr, child_path, log)

        return attr


def _build_recording_env(mock_name: str) -> tuple[dict[str, object], _CallLog]:
    log = _CallLog()
    env = _build_env(mock_name)
    wrapped: dict[str, object] = {}
    for var_name, obj in env.items():
        if callable(obj) and isinstance(obj, type):
            original_cls = obj

            def _factory(
                *a: Any, _cls: type = original_cls, _n: str = var_name, _l: _CallLog = log, **kw: Any,
            ) -> _RecordingProxy:
                return _RecordingProxy(_cls(*a, **kw), _n, _l)

            wrapped[var_name] = _factory
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            wrapped[var_name] = _RecordingProxy(obj, var_name, log)
        else:
            wrapped[var_name] = obj
    return wrapped, log


def _build_safe_builtins(env: dict[str, object]) -> dict[str, Any]:
    allowed_imports = {
        name: value
        for name, value in env.items()
        if isinstance(name, str) and name.isidentifier()
    }

    def _safe_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level != 0:
            raise ImportError("relative imports are not available in this benchmark")
        root = name.split(".", 1)[0]
        if root in allowed_imports:
            return allowed_imports[root]
        raise ImportError(f"module {name!r} is not available in this benchmark")

    builtins = dict(_SAFE_BUILTINS)
    builtins["__import__"] = _safe_import
    return builtins


def _extract_candidate(text: Any) -> str:
    if isinstance(text, (dict, list, tuple, int, float, bool)) or text is None:
        text = json.dumps(text, sort_keys=True, ensure_ascii=False)
    elif not isinstance(text, str):
        text = str(text)

    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            return parts[1].replace("python", "", 1).replace("json", "", 1).strip()
    return stripped


def _json_dumps(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _hash_trace(records: list[ActionRecord]) -> str:
    payload = [
        {
            "turn_index": record.turn_index,
            "action": asdict(record.action),
            "observation": record.observation,
        }
        for record in records
    ]
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _last_action_signature(result: EpisodeResult) -> str:
    if not result.action_records:
        return ""
    action = result.action_records[-1].action
    for value in (action.content, action.candidate, action.model):
        if value is not None:
            return _extract_candidate(value)
    return _extract_candidate(action.raw_response)


def _add_failure_tag(result: EpisodeResult, tag: str) -> EpisodeResult:
    if tag in result.failure_tags:
        return result
    return replace(result, failure_tags=tuple(sorted(set(result.failure_tags + (tag,)))))


def _looks_like_environment_rejection(candidate: object) -> bool:
    text = _extract_candidate(candidate).lower()
    if not text:
        return False
    rejection_markers = (
        "prompt injection",
        "no local docs",
        "no local registry",
        "no prior context",
        "no workspace context",
        "cannot fabricate",
        "can't fabricate",
        "won't fabricate",
        "please provide the local registry",
        "please provide the local docs",
        "no tools to search local files",
        "there is no such local",
    )
    return any(marker in text for marker in rejection_markers)


def _score_documents(query: str, scenario: Scenario) -> list[tuple[int, str, str, str]]:
    query_terms = {
        token
        for token in re.split(r"[^A-Za-z0-9_\u0080-\uffff.-]+", query.lower())
        if token
    }
    scored: list[tuple[int, str, str, str]] = []
    for doc in scenario.docs_index:
        haystack = f"{doc.title} {doc.text} {' '.join(doc.keywords)}".lower()
        score = sum(1 for token in query_terms if token in haystack)
        scored.append((score, doc.doc_id, doc.title, doc.snippet))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[:3]


def _build_env(mock_name: str) -> dict[str, object]:
    if mock_name == "openai_client":
        class CreateAPI:
            def create(self, *, model: str, input: str):
                return {"ok": True, "model": model, "input": input}

        class LegacyResponses:
            def create(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "client.responses.create was removed in SDK 3.1; use client.unified.responses.create"
                )

        class UnifiedNamespace:
            def __init__(self):
                self.responses = CreateAPI()

        class OpenAI:
            def __init__(self):
                self.unified = UnifiedNamespace()
                self.responses = LegacyResponses()

        return {"OpenAI": OpenAI, "client": OpenAI()}

    if mock_name == "openai_client_content":
        class CreateAPI:
            def create(self, *, model: str, content: str):
                return {"ok": True, "model": model, "content": content}

        class LegacyResponses:
            def create(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "client.responses.create was removed; use client.unified.responses.create with content=..."
                )

        class UnifiedNamespace:
            def __init__(self):
                self.responses = CreateAPI()

        class OpenAI:
            def __init__(self):
                self.unified = UnifiedNamespace()
                self.responses = LegacyResponses()

        return {"OpenAI": OpenAI, "client": OpenAI()}

    if mock_name == "openai_han":
        class 创建接口:
            def 创建(self, *, 模型: str, 内容: str):
                return {"ok": True, "模型": 模型, "内容": 内容}

        class 响应命名空间:
            def __init__(self):
                self.创建 = 创建接口().创建

        class 统一命名空间:
            def __init__(self):
                self.响应 = 响应命名空间()

        class OpenAIHan:
            def __init__(self):
                self.统一 = 统一命名空间()

        return {"OpenAIHan": OpenAIHan, "client": OpenAIHan()}

    if mock_name == "torch_amp":
        class AmpNamespace:
            def scope(self, device: str = "cpu", dtype: str = "float16"):
                return {"device": device, "dtype": dtype}

        class LegacyCudaAmp:
            def autocast(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "torch.cuda.amp.autocast is deprecated; use torch.amp.scope(device='cuda', dtype='float16')"
                )

        class LegacyAmpNamespace:
            def autocast(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "torch.amp.autocast is not valid in this local benchmark; use torch.amp.scope(...)"
                )

        class CudaNamespace:
            def __init__(self):
                self.amp = LegacyCudaAmp()

        class TorchNamespace:
            def __init__(self):
                self.amp = AmpNamespace()
                self.cuda = CudaNamespace()
                self.amp.autocast = LegacyAmpNamespace().autocast

        return {"torch": TorchNamespace()}

    if mock_name == "pandas_cn":
        class PandasCN:
            def concat_rows(self, frames, ignore_index: bool = False):
                return {"frame_count": len(frames), "ignore_index": ignore_index}

        return {"pandas_cn": PandasCN(), "left": object(), "right": object()}

    if mock_name == "pandas_cn_stack":
        class PandasCN:
            def stack_rows(self, frames, reset_index: bool = False):
                return {"frame_count": len(frames), "reset_index": reset_index}

            def concat_rows(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "pandas_cn.concat_rows was renamed; use pandas_cn.stack_rows(..., reset_index=True)"
                )

        return {"pandas_cn": PandasCN(), "left": object(), "right": object()}

    if mock_name == "pandas_han":
        class 中文表格层:
            def 合并行(self, frames, 忽略索引: bool = False):
                return {"frame_count": len(frames), "忽略索引": 忽略索引}

        return {"表格层": 中文表格层(), "left": object(), "right": object()}

    if mock_name == "openai_zh":
        class CreateAPI:
            def create(self, *, model: str, input: str):
                return {"ok": True, "model": model, "input": input}

        class WanchengNamespace:
            def __init__(self):
                self.create = CreateAPI().create

        class DuihuaNamespace:
            def __init__(self):
                self.wancheng = WanchengNamespace()

        class LegacyChatCompletions:
            def create(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "Legacy OpenAI path names are not valid inside OpenAIZH. Use client.duihua.wancheng.create"
                )

        class ChatNamespace:
            def __init__(self):
                self.completions = LegacyChatCompletions()

        class OpenAIZH:
            def __init__(self):
                self.duihua = DuihuaNamespace()
                self.chat = ChatNamespace()
                self.responses = LegacyChatCompletions()

        return {"OpenAIZH": OpenAIZH, "client": OpenAIZH()}

    raise BenchmarkValidationError(f"Unknown mock environment: {mock_name}")


def _execute_python(candidate: str, mock_name: str) -> tuple[bool, str, _CallLog]:
    payload = _extract_candidate(candidate)
    env, log = _build_recording_env(mock_name)
    safe_builtins = _build_safe_builtins(env)
    try:
        compiled = compile(payload, "<candidate>", "eval")
        eval(compiled, {"__builtins__": safe_builtins}, env)
    except SyntaxError:
        try:
            compiled = compile(payload, "<candidate>", "exec")
            exec(compiled, {"__builtins__": safe_builtins}, env)
        except Exception as exc:  # noqa: BLE001
            return False, f"runtime error: {exc}", log
    except Exception as exc:  # noqa: BLE001
        return False, f"runtime error: {exc}", log
    return True, "candidate executed successfully", log


def _recorded_call_matches(call: RecordedCall, expected: dict[str, Any]) -> bool:
    if call.path != expected.get("path"):
        return False
    expected_kwargs = expected.get("kwargs")
    if expected_kwargs is not None:
        for key, value in expected_kwargs.items():
            if call.kwargs.get(key) != value:
                return False
    return True


def _has_forbidden_recorded_call(log: _CallLog, forbidden_paths: tuple[str, ...]) -> bool:
    if not forbidden_paths:
        return False
    return any(call.path in forbidden_paths for call in log.calls)


def _validate_registry_schema(
    payload: dict[str, Any],
    schema_fields: tuple[str, ...],
) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload must be a JSON object"
    if len(payload) != 1:
        return False, f"expected exactly 1 field, got {len(payload)}"
    key = next(iter(payload))
    if key not in schema_fields:
        return False, f"unexpected field '{key}'"
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        return False, f"field '{key}' must be a non-empty string"
    return True, ""


def _extract_registry_payload(
    candidate_text: str,
    schema_fields: tuple[str, ...],
    *,
    allow_raw_model: bool,
) -> tuple[dict[str, Any] | None, str]:
    stripped = candidate_text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        valid, error = _validate_registry_schema(payload, schema_fields)
        if not valid:
            return None, error
        return payload, ""

    if allow_raw_model and stripped and " " not in stripped and len(stripped) <= 64:
        return {schema_fields[0]: stripped}, ""

    return None, "model selection could not be parsed"


def _validate_candidate(
    scenario: Scenario,
    candidate: str,
    *,
    allow_raw_model: bool = False,
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...], _CallLog | None, bool]:
    """Validate a candidate answer.

    Returns (functional_ok, feedback, tags, entities, call_log, semantic_ok).
    """
    spec = scenario.validator_spec
    candidate_text = _extract_candidate(candidate)
    tags: list[str] = []
    selected_entities: list[str] = []

    if spec.kind is ValidatorKind.PYTHON_MOCK:
        valid_runtime, feedback, call_log = _execute_python(candidate_text, spec.payload["mock"])
        if not valid_runtime:
            # Even on runtime error, check if expected call was attempted (for semantic credit)
            expected_call = spec.payload.get("expected_call")
            semantic_ok = False
            if expected_call is not None:
                semantic_ok = any(
                    _recorded_call_matches(c, expected_call)
                    for c in call_log.calls
                    if not c.raised
                )
            return False, feedback, ("runtime_error",), tuple(selected_entities), call_log, semantic_ok

        expected_call = spec.payload.get("expected_call")
        if expected_call is not None:
            successful_calls = [c for c in call_log.calls if not c.raised]
            if not any(_recorded_call_matches(c, expected_call) for c in successful_calls):
                tags.append("semantic_mismatch")
        elif expected_call is None:
            missing_required = [token for token in spec.required_tokens if token not in candidate_text]
            if missing_required:
                tags.append("missing_required_tokens")

        forbidden_paths = tuple(spec.payload.get("forbidden_call_paths", ()))
        if forbidden_paths:
            if _has_forbidden_recorded_call(call_log, forbidden_paths):
                tags.append("legacy_tokens_present")
        else:
            forbidden_found = [token for token in spec.forbidden_tokens if token in candidate_text]
            if forbidden_found:
                tags.append("legacy_tokens_present")

        semantic_ok = (
            "runtime_error" not in tags
            and "semantic_mismatch" not in tags
            and "missing_required_tokens" not in tags
        )
        return not tags, feedback, tuple(tags), tuple(selected_entities), call_log, semantic_ok

    # MODEL_REGISTRY path
    schema_fields = tuple(spec.payload.get("schema_fields", ("model",)))
    parsed_payload, parse_error = _extract_registry_payload(
        candidate_text,
        schema_fields,
        allow_raw_model=allow_raw_model,
    )
    expected_model = spec.payload["expected_model"]
    if parsed_payload is None:
        semantic_ok = expected_model in candidate_text
        return False, parse_error, ("model_parse_error",), tuple(selected_entities), None, semantic_ok

    field_name = next(field for field in schema_fields if field in parsed_payload)
    selected_model = parsed_payload[field_name]
    selected_entities.append(selected_model)
    registry = spec.payload["registry"]
    if selected_model not in registry:
        semantic_ok = selected_model == expected_model
        return False, f"{selected_model} is not registered locally", ("unknown_model",), tuple(selected_entities), None, semantic_ok
    if not registry[selected_model]["available"]:
        semantic_ok = selected_model == expected_model
        return False, f"{selected_model} is not available in the local registry", ("model_unavailable",), tuple(selected_entities), None, semantic_ok
    forbidden_found = [token for token in spec.forbidden_tokens if token in candidate_text]
    if forbidden_found:
        tags.append("legacy_tokens_present")
    if selected_model != expected_model:
        tags.append("wrong_model")
    semantic_ok = selected_model == expected_model
    return not tags, f"selected model {selected_model} is accepted", tuple(tags), tuple(selected_entities), None, semantic_ok


def _record(records: list[ActionRecord], turn_index: int, action: EpisodeAction, observation: str) -> None:
    records.append(ActionRecord(turn_index=turn_index, action=action, observation=observation))


def _primary_trial_action(scenario: Scenario) -> str:
    if scenario.validator_spec.kind == ValidatorKind.MODEL_REGISTRY:
        return "select_model"
    return "run_candidate"


def _primary_trial_action_spec(scenario: Scenario) -> str:
    return _ALL_ACTION_SPECS[2] if _primary_trial_action(scenario) == "run_candidate" else _ALL_ACTION_SPECS[3]


def _render_allowed_actions(action_specs: tuple[str, ...], *, label: str = "Allowed actions:") -> str:
    return label + "\n" + "\n".join(action_specs) + "\n"


def _feedback_only_action_specs(
    scenario: Scenario,
    *,
    docs_unlocked: bool,
    answer_unlocked: bool,
) -> tuple[str, ...]:
    specs: list[str] = []
    if docs_unlocked:
        specs.extend(_ALL_ACTION_SPECS[:2])
    specs.append(_primary_trial_action_spec(scenario))
    if answer_unlocked:
        specs.append(_ALL_ACTION_SPECS[4])
    return tuple(specs)


def _transfer_only_action_specs(scenario: Scenario) -> tuple[str, ...]:
    return (_primary_trial_action_spec(scenario), _ALL_ACTION_SPECS[4])


def _feedback_only_blocked_observation(
    scenario: Scenario,
    action_name: str,
    *,
    trial_taken: bool,
    docs_unlocked: bool,
    answer_unlocked: bool,
) -> str:
    primary_action = _primary_trial_action(scenario)
    if not trial_taken:
        if action_name == "answer":
            return f"Submit a trial with {primary_action} before giving a final answer. Local docs are not available yet."
        if action_name in {"search_docs", "read_doc"}:
            return f"Local docs are not available yet. Submit a trial with {primary_action} first and use the runtime feedback to adapt."
        return f"Use {primary_action} to try one candidate first."

    if action_name in {"search_docs", "read_doc"} and not docs_unlocked:
        if answer_unlocked:
            return "Local docs are still unavailable because your trial has not failed. You can answer now or try another candidate."
        return "Local docs are still unavailable. Try another candidate first."

    if action_name == "answer" and not answer_unlocked:
        return f"Submit a trial with {primary_action} before giving a final answer."

    return "Use one of the currently allowed actions."


def _transfer_only_blocked_observation(scenario: Scenario, action_name: str) -> str:
    if action_name in {"search_docs", "read_doc"}:
        return "Local docs are not available in this transfer stage. Use what you already learned, or try another candidate."
    return f"Use {_primary_trial_action(scenario)} or answer with your final output."


def _feedback_only_failure_observation(observation: str) -> str:
    return observation + "\nLocal docs are now available through search_local_docs and read_local_doc."


def _render_scenario_prompt(scenario: Scenario, prompt_style: PromptStyle) -> str:
    if prompt_style == "release_note":
        task = scenario.prompt
        task = task.replace(
            "You are solving an adaptive interface benchmark for ",
            "I'm working in a local integration workspace for ",
        )
        task = task.replace(
            "Local benchmark evidence overrides your prior knowledge.\n",
            "I just released or updated the local interface here. Use the local information in this workspace instead of assuming older defaults.\n",
        )
        task = task.replace(
            "The latest docs are available through the local search_docs and read_doc actions.",
            "If you need docs, use the local doc search and local doc read actions in this workspace.",
        )
        task = task.replace(
            "No documentation is provided up front. Use runtime feedback to adapt.",
            "I did not include docs up front. If something fails, use the runtime feedback to adapt.",
        )
        task = task.replace(
            "No documentation is available in this transfer stage. Use what you learned earlier in the local workspace.",
            "I did not include docs in this transfer stage. Use what you learned earlier in this local workspace.",
        )
        task = task.replace(
            "The local change is:",
            "I just released this local change:",
        )
        return task
    return scenario.prompt


def _initial_prompt(scenario: Scenario, prompt_style: PromptStyle) -> str:
    strict_feedback_only = scenario.evidence_regime == EvidenceRegime.FEEDBACK_ONLY
    transfer_only = scenario.evidence_regime == EvidenceRegime.TRANSFER_ONLY
    if strict_feedback_only:
        action_specs = _feedback_only_action_specs(
            scenario,
            docs_unlocked=False,
            answer_unlocked=False,
        )
    elif transfer_only:
        action_specs = _transfer_only_action_specs(scenario)
    else:
        action_specs = _ALL_ACTION_SPECS

    if prompt_style == "release_note":
        if strict_feedback_only:
            intro = (
                "You are helping with a local software integration task.\n"
                "There is no internet access and no live web search.\n"
                "Some package versions or model names may be newly released in this local environment.\n"
                "Start by trying one candidate based on your current best guess. Use the resulting runtime feedback to adapt.\n"
                "Respond with JSON only.\n"
            )
        elif transfer_only:
            intro = (
                "You are helping with a local software integration task.\n"
                "There is no internet access and no live web search.\n"
                "No local docs are available in this transfer stage. Use what you already learned earlier in this workspace.\n"
                "Respond with JSON only.\n"
            )
        else:
            intro = (
                "You are helping with a local software integration task.\n"
                "There is no internet access and no live web search.\n"
                "If you search, you are only searching local docs that came with this workspace.\n"
                "Some package versions or model names may be newly released in this local environment.\n"
                "Respond with JSON only.\n"
            )
    else:
        if strict_feedback_only:
            intro = (
                "You are inside an offline adaptive interface benchmark.\n"
                "Local benchmark evidence overrides your prior knowledge.\n"
                "There is no internet access and no live web search.\n"
                "Start by trying one candidate based on your current best guess. Use the resulting runtime feedback to adapt.\n"
                "Respond with JSON only.\n"
            )
        elif transfer_only:
            intro = (
                "You are inside an offline adaptive interface benchmark.\n"
                "Local benchmark evidence overrides your prior knowledge.\n"
                "There is no internet access and no live web search.\n"
                "No local docs are available in this transfer stage. Use what you already learned earlier in this benchmark.\n"
                "Respond with JSON only.\n"
            )
        else:
            intro = (
                "You are inside an offline adaptive interface benchmark.\n"
                "Local benchmark evidence overrides your prior knowledge.\n"
                "There is no internet access and no live web search. Any search action only queries local benchmark sources.\n"
                "Respond with JSON only.\n"
            )
    return (
        intro
        + _render_allowed_actions(action_specs)
        + f"Task:\n{_render_scenario_prompt(scenario, prompt_style)}"
    )


def _followup_prompt(
    observation: str,
    prompt_style: PromptStyle,
    *,
    allowed_actions: tuple[str, ...] | None = None,
) -> str:
    prefix = "Local observation:\n" if prompt_style == "release_note" else "Observation:\n"
    parts = [f"{prefix}{observation}"]
    if allowed_actions is not None:
        parts.append(_render_allowed_actions(allowed_actions, label="Allowed actions now:").rstrip())
    parts.append("Decide the next action. Respond with JSON only.")
    return "\n".join(parts)


def _compute_efficiency(turn_count: int, max_turns: int, ideal_turns: int) -> float:
    if turn_count <= ideal_turns:
        return 1.0
    denominator = max(1, max_turns - ideal_turns)
    return max(0.0, 1.0 - ((turn_count - ideal_turns) / denominator))


def run_scenario(
    adapter: LLMAdapter,
    scenario: Scenario,
    attempt_index: int = 0,
    prompt_style: PromptStyle = "benchmark",
    *,
    reset_adapter: bool = True,
) -> EpisodeResult:
    if reset_adapter:
        adapter.reset()
    records: list[ActionRecord] = []
    used_docs: list[str] = []
    surfaced_docs: list[str] = []
    selected_entities: list[str] = []
    failure_tags: list[str] = []
    saw_feedback = False
    final_candidate: str | None = None
    strict_feedback_only = scenario.evidence_regime == EvidenceRegime.FEEDBACK_ONLY
    transfer_only = scenario.evidence_regime == EvidenceRegime.TRANSFER_ONLY
    trial_taken = False
    docs_unlocked = not (strict_feedback_only or transfer_only)
    answer_unlocked = not strict_feedback_only
    protocol_violations = 0
    evidence_action_count = 0
    deadline = time.monotonic() + scenario.timeout_s
    prompt = _initial_prompt(scenario, prompt_style)

    for turn_index in range(scenario.max_turns):
        if time.monotonic() > deadline:
            failure_tags.append("timeout")
            break

        raw_response = adapter.prompt(prompt)
        action = parse_action(raw_response)
        if strict_feedback_only or transfer_only:
            allowed_actions = {_primary_trial_action(scenario)}
            if strict_feedback_only and docs_unlocked:
                allowed_actions.update({"search_docs", "read_doc"})
            if answer_unlocked:
                allowed_actions.add("answer")
            if action.action not in allowed_actions:
                protocol_violations += 1
                if transfer_only:
                    observation = _transfer_only_blocked_observation(scenario, action.action)
                    allowed_action_specs = _transfer_only_action_specs(scenario)
                else:
                    observation = _feedback_only_blocked_observation(
                        scenario,
                        action.action,
                        trial_taken=trial_taken,
                        docs_unlocked=docs_unlocked,
                        answer_unlocked=answer_unlocked,
                    )
                    allowed_action_specs = _feedback_only_action_specs(
                        scenario,
                        docs_unlocked=docs_unlocked,
                        answer_unlocked=answer_unlocked,
                    )
                _record(records, turn_index, action, observation)
                prompt = _followup_prompt(
                    observation,
                    prompt_style,
                    allowed_actions=allowed_action_specs,
                )
                continue

        if action.action == "search_docs":
            evidence_action_count += 1
            query = action.query or ""
            results = _score_documents(query, scenario)
            surfaced_docs.extend([doc_id for score, doc_id, _, _ in results if score > 0])
            observation = _json_dumps(
                [
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "snippet": snippet,
                    }
                    for _, doc_id, title, snippet in results
                ]
            )
            _record(records, turn_index, action, observation)
            prompt = _followup_prompt(
                observation,
                prompt_style,
                allowed_actions=(
                    _feedback_only_action_specs(
                        scenario,
                        docs_unlocked=docs_unlocked,
                        answer_unlocked=answer_unlocked,
                    )
                    if strict_feedback_only
                    else None
                ),
            )
            continue

        if action.action == "read_doc":
            evidence_action_count += 1
            matching = next((doc for doc in scenario.docs_index if doc.doc_id == action.doc_id), None)
            if matching is None:
                observation = f"doc {action.doc_id!r} not found"
            else:
                used_docs.append(matching.doc_id)
                observation = matching.text
            _record(records, turn_index, action, observation)
            prompt = _followup_prompt(
                observation,
                prompt_style,
                allowed_actions=(
                    _feedback_only_action_specs(
                        scenario,
                        docs_unlocked=docs_unlocked,
                        answer_unlocked=answer_unlocked,
                    )
                    if strict_feedback_only
                    else None
                ),
            )
            continue

        if action.action in {"run_candidate", "select_model"}:
            candidate = action.candidate or action.model or action.content or ""
            valid, observation, tags, entities, _, _ = _validate_candidate(
                scenario,
                candidate,
                allow_raw_model=action.action == "select_model",
            )
            saw_feedback = True
            trial_taken = True
            selected_entities.extend(entities)
            if not valid:
                failure_tags.extend(tags)
                if strict_feedback_only:
                    docs_unlocked = True
                    answer_unlocked = True
                    observation = _feedback_only_failure_observation(observation)
            elif strict_feedback_only:
                answer_unlocked = True
            final_candidate = candidate
            _record(records, turn_index, action, observation)
            prompt = _followup_prompt(
                observation,
                prompt_style,
                allowed_actions=(
                    _feedback_only_action_specs(
                        scenario,
                        docs_unlocked=docs_unlocked,
                        answer_unlocked=answer_unlocked,
                    )
                    if strict_feedback_only
                    else None
                ),
            )
            continue

        if action.action == "answer":
            final_candidate = action.content or action.candidate or action.model or action.raw_response
            _record(records, turn_index, action, "final answer submitted")
            break

        protocol_violations += 1
        failure_tags.append("unsupported_action")
        _record(records, turn_index, action, "unsupported action")
        prompt = _followup_prompt(
            "Unsupported action. Use one of the allowed actions.",
            prompt_style,
        )

    if final_candidate is None:
        protocol_violations += 1
        failure_tags.append("no_final_answer")
        final_candidate = ""

    functional_correctness, _, validation_tags, entities, final_log, semantic_ok = _validate_candidate(
        scenario,
        final_candidate,
        allow_raw_model=False,
    )
    selected_entities.extend(entities)
    failure_tags.extend(validation_tags)
    if any(tag in validation_tags for tag in {"model_parse_error"}):
        protocol_violations += 1

    adaptation_correctness = True
    if scenario.validator_spec.accepted_evidence_doc_ids:
        accepted_docs = set(scenario.validator_spec.accepted_evidence_doc_ids)
        observed_docs = set(used_docs).union(surfaced_docs)
        if not accepted_docs.intersection(observed_docs):
            adaptation_correctness = False
            failure_tags.append("missing_required_docs")
    elif scenario.validator_spec.must_use_doc_ids:
        required_docs = set(scenario.validator_spec.must_use_doc_ids)
        if not required_docs.issubset(set(used_docs)):
            adaptation_correctness = False
            failure_tags.append("missing_required_docs")
    if scenario.validator_spec.must_observe_feedback and not saw_feedback:
        adaptation_correctness = False
        failure_tags.append("missing_feedback_cycle")
    if not functional_correctness:
        adaptation_correctness = False

    candidate_text = _extract_candidate(final_candidate)
    forbidden_paths = tuple(scenario.validator_spec.payload.get("forbidden_call_paths", ()))
    if final_log is not None and forbidden_paths:
        legacy_avoidance = not _has_forbidden_recorded_call(final_log, forbidden_paths)
    else:
        legacy_avoidance = not any(token in candidate_text for token in scenario.validator_spec.forbidden_tokens)
    if not legacy_avoidance:
        failure_tags.append("legacy_final_answer")
    if (
        scenario.evidence_regime in {
            EvidenceRegime.EXPLICIT_CHANGE,
            EvidenceRegime.DOCS_INLINE,
            EvidenceRegime.DOCS_SEARCH,
            EvidenceRegime.DEPRECATION_NOTICE,
        }
        and evidence_action_count == 0
        and not semantic_ok
        and _looks_like_environment_rejection(final_candidate)
    ):
        failure_tags.append("environment_rejected")

    turn_count = len(records)
    semantic_correctness = float(semantic_ok)
    protocol_compliance = max(0.0, 1.0 - (protocol_violations / max(1, scenario.max_turns)))
    ideal_turns = scenario.ideal_turns or (2 if strict_feedback_only else 1)
    efficiency_score = _compute_efficiency(turn_count, scenario.max_turns, ideal_turns)

    if scenario.sequence_id is None:
        score = (
            0.60 * semantic_correctness
            + 0.25 * float(adaptation_correctness)
            + 0.15 * float(legacy_avoidance)
        )
        passed = functional_correctness and adaptation_correctness and legacy_avoidance
    else:
        score = (
            0.55 * semantic_correctness
            + 0.20 * float(adaptation_correctness)
            + 0.15 * protocol_compliance
            + 0.10 * efficiency_score
        )
        passed = semantic_correctness == 1.0 and adaptation_correctness and protocol_compliance > 0.0

    deduped_tags = tuple(sorted(set(failure_tags)))
    deduped_entities = tuple(sorted(set(entity for entity in selected_entities if entity)))
    trace_digest = _hash_trace(records)
    score_breakdown = {
        "semantic_correctness": semantic_correctness,
        "adaptation_correctness": float(adaptation_correctness),
        "legacy_avoidance": float(legacy_avoidance),
        "protocol_compliance": protocol_compliance,
        "efficiency_score": efficiency_score,
    }
    return EpisodeResult(
        scenario_id=scenario.id,
        family=scenario.family,
        attempt_index=attempt_index,
        passed=passed,
        score=score,
        failure_tags=deduped_tags,
        selected_entities=deduped_entities,
        used_evidence_ids=tuple(sorted(set(used_docs))),
        surfaced_evidence_ids=tuple(sorted(set(surfaced_docs))),
        trace_digest=trace_digest,
        action_records=tuple(records),
        functional_correctness=functional_correctness,
        adaptation_correctness=adaptation_correctness,
        legacy_avoidance=legacy_avoidance,
        semantic_correctness=semantic_correctness,
        protocol_compliance=protocol_compliance,
        efficiency_score=efficiency_score,
        learning_transfer_score=0.0,
        turn_count=turn_count,
        evidence_action_count=evidence_action_count,
        benchmark_suite=scenario.benchmark_suite,
        sequence_id=scenario.sequence_id,
        sequence_stage=scenario.sequence_stage,
        score_breakdown=score_breakdown,
    )


def run_sequence(
    adapter: LLMAdapter,
    sequence: ScenarioSequence,
    attempt_index: int = 0,
    prompt_style: PromptStyle = "release_note",
) -> SequenceResult:
    adapter.reset()
    stage_results: list[EpisodeResult] = []
    learned_rules: set[str] = set()
    transfer_scores: list[float] = []
    benchmark_suite = sequence.benchmark_suite

    for stage in sequence.stages:
        learned_before_stage = set(learned_rules)
        result = run_scenario(
            adapter,
            stage,
            attempt_index=attempt_index,
            prompt_style=prompt_style,
            reset_adapter=False,
        )
        transfer_score = 0.0
        if stage.depends_on_rules:
            overlap = len(learned_before_stage.intersection(stage.depends_on_rules))
            if result.semantic_correctness >= 1.0 and result.adaptation_correctness:
                transfer_score = overlap / max(1, len(stage.depends_on_rules))
            transfer_scores.append(transfer_score)
        if result.semantic_correctness >= 1.0 and result.adaptation_correctness:
            learned_rules.update(stage.teaches_rules)
        enriched_breakdown = dict(result.score_breakdown)
        enriched_breakdown["learning_transfer_score"] = transfer_score
        stage_results.append(
            replace(
                result,
                learning_transfer_score=transfer_score,
                score_breakdown=enriched_breakdown,
            )
        )

    prior_probe_correctness = 0.0
    stale_prior_rate = 0.0
    revision_success = 0.0
    revision_turns_to_fix = 0.0
    revision_efficiency = 0.0
    transfer_after_revision = 0.0
    localized_generalization = 0.0
    learning_score = 0.0

    if benchmark_suite == "v2_learning" and len(stage_results) >= 4:
        prior = stage_results[0]
        revision = stage_results[1]
        transfer = stage_results[2]
        localized = stage_results[3]

        if prior.semantic_correctness < 1.0:
            prior = _add_failure_tag(prior, "stale_prior_guess")

        prior_signature = _last_action_signature(prior)
        revision_signature = _last_action_signature(revision)
        output_changed = revision_signature != prior_signature

        if prior.semantic_correctness < 1.0 and revision.semantic_correctness < 1.0:
            if revision.evidence_action_count > 0:
                if output_changed:
                    revision = _add_failure_tag(revision, "updated_to_wrong_local_rule")
                else:
                    revision = _add_failure_tag(revision, "searched_but_not_updated")
            else:
                revision = _add_failure_tag(revision, "stale_prior_persisted")

        if localized.semantic_correctness < 1.0:
            localized = _add_failure_tag(localized, "localized_surface_gap")

        normalized_results: list[EpisodeResult] = []
        for result in (prior, revision, transfer, localized, *stage_results[4:]):
            if result.semantic_correctness >= 1.0 and (
                not result.adaptation_correctness or result.protocol_compliance <= 0.0
            ):
                result = _add_failure_tag(result, "format_only_failure")
            normalized_results.append(result)
        stage_results = normalized_results

        prior_probe_correctness = stage_results[0].semantic_correctness
        stale_prior_rate = 1.0 - prior_probe_correctness
        revision_output_changed = _last_action_signature(stage_results[1]) != _last_action_signature(stage_results[0])
        revision_success = float(
            stage_results[1].passed
            and (stage_results[0].semantic_correctness >= 1.0 or revision_output_changed)
        )
        revision_turns_to_fix = float(stage_results[1].turn_count if revision_success else 0.0)
        revision_efficiency = stage_results[1].efficiency_score if revision_success else 0.0
        transfer_after_revision = float(revision_success and stage_results[2].passed)
        localized_generalization = float(
            stage_results[3].semantic_correctness >= 1.0 and stage_results[3].protocol_compliance > 0.0
        )
        learning_score = (
            V2_LEARNING_SCORE_WEIGHTS["revision_success"] * revision_success
            + V2_LEARNING_SCORE_WEIGHTS["transfer_after_revision"] * transfer_after_revision
            + V2_LEARNING_SCORE_WEIGHTS["localized_generalization"] * localized_generalization
        )

    in_task_adaptation = sum(float(result.adaptation_correctness) for result in stage_results) / max(1, len(stage_results))
    cross_task_transfer = sum(transfer_scores) / max(1, len(transfer_scores)) if transfer_scores else 0.0
    semantic_correctness = sum(result.semantic_correctness for result in stage_results) / max(1, len(stage_results))
    protocol_compliance = sum(result.protocol_compliance for result in stage_results) / max(1, len(stage_results))
    efficiency_score = sum(result.efficiency_score for result in stage_results) / max(1, len(stage_results))
    if benchmark_suite == "v2_learning":
        overall_score = learning_score
        passed = revision_success == 1.0 and transfer_after_revision == 1.0 and localized_generalization == 1.0
    else:
        overall_score = (
            0.35 * semantic_correctness
            + 0.25 * in_task_adaptation
            + 0.25 * cross_task_transfer
            + 0.10 * protocol_compliance
            + 0.05 * efficiency_score
        )
        passed = all(result.passed for result in stage_results) and cross_task_transfer >= 1.0
    score_breakdown = {
        "semantic_correctness": semantic_correctness,
        "in_task_adaptation": in_task_adaptation,
        "cross_task_transfer": cross_task_transfer,
        "protocol_compliance": protocol_compliance,
        "efficiency_score": efficiency_score,
    }
    if benchmark_suite == "v2_learning":
        score_breakdown.update(
            {
                "prior_probe_correctness": prior_probe_correctness,
                "stale_prior_rate": stale_prior_rate,
                "revision_success": revision_success,
                "revision_turns_to_fix": revision_turns_to_fix,
                "revision_efficiency": revision_efficiency,
                "transfer_after_revision": transfer_after_revision,
                "localized_generalization": localized_generalization,
                "learning_score": learning_score,
            }
        )
    return SequenceResult(
        sequence_id=sequence.id,
        family=sequence.family,
        attempt_index=attempt_index,
        passed=passed,
        overall_score=overall_score,
        stage_results=tuple(stage_results),
        in_task_adaptation=in_task_adaptation,
        cross_task_transfer=cross_task_transfer,
        semantic_correctness=semantic_correctness,
        protocol_compliance=protocol_compliance,
        efficiency_score=efficiency_score,
        benchmark_suite=benchmark_suite,
        learned_rules=tuple(sorted(learned_rules)),
        prior_probe_correctness=prior_probe_correctness,
        stale_prior_rate=stale_prior_rate,
        revision_success=revision_success,
        revision_turns_to_fix=revision_turns_to_fix,
        revision_efficiency=revision_efficiency,
        transfer_after_revision=transfer_after_revision,
        localized_generalization=localized_generalization,
        learning_score=learning_score,
        score_breakdown=score_breakdown,
    )


def run_suite(
    adapter_factory: Callable[[Scenario, int], LLMAdapter],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    include_stress: bool = False,
) -> dict[str, object]:
    scenarios = list(build_core_suite())
    if include_stress:
        scenarios.extend(build_stress_suite())

    results: list[EpisodeResult] = []
    for scenario in scenarios:
        for attempt_index in range(attempts):
            adapter = adapter_factory(scenario, attempt_index)
            results.append(run_scenario(adapter, scenario, attempt_index=attempt_index))
    return aggregate_attempts(results)


def run_sequence_suite(
    adapter_factory: Callable[[ScenarioSequence, int], LLMAdapter],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    prompt_style: PromptStyle = "release_note",
    suite: str = "v2",
) -> dict[str, object]:
    if suite == "v2_learning":
        sequences = build_v2_learning_sequences()
    else:
        sequences = build_v2_sequences()
    results: list[SequenceResult] = []
    for sequence in sequences:
        for attempt_index in range(attempts):
            adapter = adapter_factory(sequence, attempt_index)
            results.append(
                run_sequence(
                    adapter,
                    sequence,
                    attempt_index=attempt_index,
                    prompt_style=prompt_style,
                )
            )
    return aggregate_sequence_results(results)
