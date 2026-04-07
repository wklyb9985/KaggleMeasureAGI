from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ScenarioFamily(StrEnum):
    API_MIGRATION = "api_migration"
    DSL_WRAPPER = "dsl_wrapper"
    FUTURE_REGISTRY = "future_registry"


class EvidenceRegime(StrEnum):
    EXPLICIT_CHANGE = "explicit_change"
    DOCS_INLINE = "docs_inline"
    DOCS_SEARCH = "docs_search"
    FEEDBACK_ONLY = "feedback_only"
    DEPRECATION_NOTICE = "deprecation_notice"


class ValidatorKind(StrEnum):
    PYTHON_MOCK = "python_mock"
    MODEL_REGISTRY = "model_registry"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    keywords: tuple[str, ...] = ()

    @property
    def snippet(self) -> str:
        first_line = self.text.strip().splitlines()[0]
        return first_line[:160]


@dataclass(frozen=True)
class ValidatorSpec:
    kind: ValidatorKind
    required_tokens: tuple[str, ...] = ()
    forbidden_tokens: tuple[str, ...] = ()
    must_use_doc_ids: tuple[str, ...] = ()
    must_observe_feedback: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    id: str
    family: ScenarioFamily
    difficulty: str
    evidence_regime: EvidenceRegime
    prompt: str
    docs_index: tuple[Document, ...]
    tool_fixtures: dict[str, Any]
    validator_spec: ValidatorSpec
    max_turns: int
    timeout_s: int
    attempts: int = 5


@dataclass(frozen=True)
class EpisodeAction:
    action: str
    query: str | None = None
    doc_id: str | None = None
    candidate: Any = None
    model: Any = None
    content: Any = None
    raw_response: str = ""


@dataclass(frozen=True)
class ActionRecord:
    turn_index: int
    action: EpisodeAction
    observation: str


@dataclass(frozen=True)
class EpisodeResult:
    scenario_id: str
    family: ScenarioFamily
    attempt_index: int
    passed: bool
    score: float
    failure_tags: tuple[str, ...]
    selected_entities: tuple[str, ...]
    used_evidence_ids: tuple[str, ...]
    trace_digest: str
    action_records: tuple[ActionRecord, ...]
    functional_correctness: bool
    adaptation_correctness: bool
    legacy_avoidance: bool
