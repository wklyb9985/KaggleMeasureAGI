from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from typing import Any, Callable, Literal

from adaptive_shift_bench.llm import LLMAdapter
from adaptive_shift_bench.models import (
    ActionRecord,
    EpisodeAction,
    EpisodeResult,
    Scenario,
    ScenarioFamily,
    ValidatorKind,
)
from adaptive_shift_bench.parsing import parse_action
from adaptive_shift_bench.reporting import aggregate_attempts
from adaptive_shift_bench.scenarios import DEFAULT_ATTEMPTS, build_core_suite, build_stress_suite


class BenchmarkValidationError(RuntimeError):
    pass


class DeprecatedInterfaceError(RuntimeError):
    pass


class RegistrySelectionError(RuntimeError):
    pass


PromptStyle = Literal["benchmark", "release_note"]


def _extract_candidate(text: Any) -> str:
    if isinstance(text, (dict, list, tuple, int, float, bool)) or text is None:
        text = json.dumps(text, sort_keys=True, ensure_ascii=True)
    elif not isinstance(text, str):
        text = str(text)

    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            return parts[1].replace("python", "", 1).strip()
    return stripped


def _json_dumps(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True)


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


def _score_documents(query: str, scenario: Scenario) -> list[tuple[int, str, str, str]]:
    query_terms = {
        token
        for token in re.split(r"[^A-Za-z0-9_.-]+", query.lower())
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

    if mock_name == "torch_amp":
        class AmpNamespace:
            def scope(self, device: str = "cpu", dtype: str = "float16"):
                return {"device": device, "dtype": dtype}

        class LegacyCudaAmp:
            def autocast(self, *args, **kwargs):
                raise DeprecatedInterfaceError(
                    "torch.cuda.amp.autocast is deprecated; use torch.amp.scope(device='cuda', dtype='float16')"
                )

        class CudaNamespace:
            def __init__(self):
                self.amp = LegacyCudaAmp()

        class TorchNamespace:
            def __init__(self):
                self.amp = AmpNamespace()
                self.cuda = CudaNamespace()

        return {"torch": TorchNamespace()}

    if mock_name == "pandas_cn":
        class PandasCN:
            def concat_rows(self, frames, ignore_index: bool = False):
                return {"frame_count": len(frames), "ignore_index": ignore_index}

        return {"pandas_cn": PandasCN(), "left": object(), "right": object()}

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


def _execute_python(candidate: str, mock_name: str) -> tuple[bool, str]:
    payload = _extract_candidate(candidate)
    env = _build_env(mock_name)
    try:
        compiled = compile(payload, "<candidate>", "eval")
        eval(compiled, {"__builtins__": {}}, env)
    except SyntaxError:
        try:
            compiled = compile(payload, "<candidate>", "exec")
            exec(compiled, {"__builtins__": {}}, env)
        except Exception as exc:  # noqa: BLE001
            return False, f"runtime error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"runtime error: {exc}"
    return True, "candidate executed successfully"


def _extract_selected_model(candidate: str) -> str | None:
    stripped = _extract_candidate(candidate)
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict) and isinstance(payload.get("model"), str):
            return payload["model"]
    except json.JSONDecodeError:
        pass

    direct_match = re.search(r"\b([A-Za-z0-9][A-Za-z0-9._-]{2,})\b", stripped)
    if direct_match:
        return direct_match.group(1)
    return None


def _validate_candidate(scenario: Scenario, candidate: str) -> tuple[bool, str, tuple[str, ...], tuple[str, ...]]:
    spec = scenario.validator_spec
    candidate_text = _extract_candidate(candidate)
    tags: list[str] = []
    selected_entities: list[str] = []

    missing_required = [token for token in spec.required_tokens if token not in candidate_text]
    forbidden_found = [token for token in spec.forbidden_tokens if token in candidate_text]

    if spec.kind is ValidatorKind.PYTHON_MOCK:
        valid_runtime, feedback = _execute_python(candidate_text, spec.payload["mock"])
        if not valid_runtime:
            return False, feedback, ("runtime_error",), tuple(selected_entities)
        if missing_required:
            tags.append("missing_required_tokens")
        if forbidden_found:
            tags.append("legacy_tokens_present")
        return not tags, feedback, tuple(tags), tuple(selected_entities)

    selected_model = _extract_selected_model(candidate_text)
    if selected_model is not None:
        selected_entities.append(selected_model)
    registry = spec.payload["registry"]
    expected_model = spec.payload["expected_model"]
    if selected_model is None:
        return False, "model selection could not be parsed", ("model_parse_error",), tuple(selected_entities)
    if selected_model not in registry:
        return False, f"{selected_model} is not registered locally", ("unknown_model",), tuple(selected_entities)
    if not registry[selected_model]["available"]:
        return False, f"{selected_model} is not available in the local registry", ("model_unavailable",), tuple(selected_entities)
    if forbidden_found:
        tags.append("legacy_tokens_present")
    if selected_model != expected_model:
        tags.append("wrong_model")
    return not tags, f"selected model {selected_model} is accepted", tuple(tags), tuple(selected_entities)


def _record(records: list[ActionRecord], turn_index: int, action: EpisodeAction, observation: str) -> None:
    records.append(ActionRecord(turn_index=turn_index, action=action, observation=observation))


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
            "The local change is:",
            "I just released this local change:",
        )
        return task
    return scenario.prompt


def _initial_prompt(scenario: Scenario, prompt_style: PromptStyle) -> str:
    if prompt_style == "release_note":
        intro = (
            "You are helping with a local software integration task.\n"
            "There is no internet access and no live web search.\n"
            "If you search, you are only searching local docs that came with this workspace.\n"
            "Some package versions or model names may be newly released in this local environment.\n"
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
        + "Allowed actions:\n"
        + '- {"action":"search_local_docs","query":"..."}\n'
        + '- {"action":"read_local_doc","doc_id":"..."}\n'
        + '- {"action":"run_candidate","candidate":"..."}\n'
        + '- {"action":"select_model","model":"..."}\n'
        + '- {"action":"answer","content":"..."}\n'
        + f"Task:\n{_render_scenario_prompt(scenario, prompt_style)}"
    )


def _followup_prompt(observation: str, prompt_style: PromptStyle) -> str:
    prefix = (
        "Local observation:\n" if prompt_style == "release_note" else "Observation:\n"
    )
    return (
        f"{prefix}{observation}\n"
        "Decide the next action. Respond with JSON only."
    )


def run_scenario(
    adapter: LLMAdapter,
    scenario: Scenario,
    attempt_index: int = 0,
    prompt_style: PromptStyle = "benchmark",
) -> EpisodeResult:
    adapter.reset()
    records: list[ActionRecord] = []
    used_docs: list[str] = []
    selected_entities: list[str] = []
    failure_tags: list[str] = []
    saw_feedback = False
    final_candidate: str | None = None
    deadline = time.monotonic() + scenario.timeout_s
    prompt = _initial_prompt(scenario, prompt_style)

    for turn_index in range(scenario.max_turns):
        if time.monotonic() > deadline:
            failure_tags.append("timeout")
            break

        raw_response = adapter.prompt(prompt)
        action = parse_action(raw_response)

        if action.action == "search_docs":
            query = action.query or ""
            results = _score_documents(query, scenario)
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
            prompt = _followup_prompt(observation, prompt_style)
            continue

        if action.action == "read_doc":
            matching = next((doc for doc in scenario.docs_index if doc.doc_id == action.doc_id), None)
            if matching is None:
                observation = f"doc {action.doc_id!r} not found"
            else:
                used_docs.append(matching.doc_id)
                observation = matching.text
            _record(records, turn_index, action, observation)
            prompt = _followup_prompt(observation, prompt_style)
            continue

        if action.action in {"run_candidate", "select_model"}:
            candidate = action.candidate or action.model or action.content or ""
            valid, observation, tags, entities = _validate_candidate(scenario, candidate)
            saw_feedback = True
            selected_entities.extend(entities)
            if not valid:
                failure_tags.extend(tags)
            final_candidate = candidate
            _record(records, turn_index, action, observation)
            prompt = _followup_prompt(observation, prompt_style)
            continue

        if action.action == "answer":
            final_candidate = action.content or action.candidate or action.model or action.raw_response
            _record(records, turn_index, action, "final answer submitted")
            break

        failure_tags.append("unsupported_action")
        _record(records, turn_index, action, "unsupported action")
        prompt = _followup_prompt(
            "Unsupported action. Use one of the allowed actions.",
            prompt_style,
        )

    if final_candidate is None:
        failure_tags.append("no_final_answer")
        final_candidate = ""

    functional_correctness, _, validation_tags, entities = _validate_candidate(scenario, final_candidate)
    selected_entities.extend(entities)
    failure_tags.extend(validation_tags)

    adaptation_correctness = True
    if scenario.validator_spec.must_use_doc_ids:
        required_docs = set(scenario.validator_spec.must_use_doc_ids)
        if not required_docs.issubset(set(used_docs)):
            adaptation_correctness = False
            failure_tags.append("missing_required_docs")
    if scenario.validator_spec.must_observe_feedback and not saw_feedback:
        adaptation_correctness = False
        failure_tags.append("missing_feedback_cycle")
    if not functional_correctness:
        adaptation_correctness = False

    legacy_avoidance = not any(token in _extract_candidate(final_candidate) for token in scenario.validator_spec.forbidden_tokens)
    if not legacy_avoidance:
        failure_tags.append("legacy_final_answer")

    score = (
        0.60 * float(functional_correctness)
        + 0.25 * float(adaptation_correctness)
        + 0.15 * float(legacy_avoidance)
    )
    passed = functional_correctness and adaptation_correctness and legacy_avoidance

    deduped_tags = tuple(sorted(set(failure_tags)))
    deduped_entities = tuple(sorted(set(entity for entity in selected_entities if entity)))
    trace_digest = _hash_trace(records)
    return EpisodeResult(
        scenario_id=scenario.id,
        family=scenario.family,
        attempt_index=attempt_index,
        passed=passed,
        score=score,
        failure_tags=deduped_tags,
        selected_entities=deduped_entities,
        used_evidence_ids=tuple(sorted(set(used_docs))),
        trace_digest=trace_digest,
        action_records=tuple(records),
        functional_correctness=functional_correctness,
        adaptation_correctness=adaptation_correctness,
        legacy_avoidance=legacy_avoidance,
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
