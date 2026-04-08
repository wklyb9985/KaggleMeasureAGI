from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


FEEDBACK_ONLY_POLICY = "trial_first_reveal_after_fail"


class ScenarioFamily(StrEnum):
    API_MIGRATION = "api_migration"
    DSL_WRAPPER = "dsl_wrapper"
    FUTURE_REGISTRY = "future_registry"
    LOCALIZED_API = "localized_api"


class EvidenceRegime(StrEnum):
    EXPLICIT_CHANGE = "explicit_change"
    DOCS_INLINE = "docs_inline"
    DOCS_SEARCH = "docs_search"
    FEEDBACK_ONLY = "feedback_only"
    TRANSFER_ONLY = "transfer_only"
    DEPRECATION_NOTICE = "deprecation_notice"


class ValidatorKind(StrEnum):
    PYTHON_MOCK = "python_mock"
    MODEL_REGISTRY = "model_registry"


class SequenceStage(StrEnum):
    TEACH = "teach"
    ADAPT = "adapt"
    TRANSFER = "transfer"
    CAPSTONE = "capstone"


class LanguageSurface(StrEnum):
    ENGLISH = "en"
    CHINESE = "zh"


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
    accepted_evidence_doc_ids: tuple[str, ...] = ()
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
    benchmark_suite: str = "v1"
    sequence_id: str | None = None
    sequence_stage: SequenceStage | None = None
    language_surface: LanguageSurface = LanguageSurface.ENGLISH
    teaches_rules: tuple[str, ...] = ()
    depends_on_rules: tuple[str, ...] = ()
    ideal_turns: int | None = None


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
    semantic_correctness: float = 0.0
    protocol_compliance: float = 1.0
    efficiency_score: float = 1.0
    learning_transfer_score: float = 0.0
    turn_count: int = 0
    evidence_action_count: int = 0
    surfaced_evidence_ids: tuple[str, ...] = ()
    benchmark_suite: str = "v1"
    sequence_id: str | None = None
    sequence_stage: SequenceStage | None = None
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioSequence:
    id: str
    family: ScenarioFamily
    stages: tuple[Scenario, ...]
    benchmark_suite: str = "v2"


@dataclass(frozen=True)
class SequenceResult:
    sequence_id: str
    family: ScenarioFamily
    attempt_index: int
    passed: bool
    overall_score: float
    stage_results: tuple[EpisodeResult, ...]
    in_task_adaptation: float
    cross_task_transfer: float
    semantic_correctness: float
    protocol_compliance: float
    efficiency_score: float
    benchmark_suite: str = "v2"
    learned_rules: tuple[str, ...] = ()
    prior_probe_correctness: float = 0.0
    stale_prior_rate: float = 0.0
    revision_success: float = 0.0
    revision_turns_to_fix: float = 0.0
    revision_efficiency: float = 0.0
    transfer_after_revision: float = 0.0
    localized_generalization: float = 0.0
    learning_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
