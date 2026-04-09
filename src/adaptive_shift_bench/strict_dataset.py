from __future__ import annotations

from functools import lru_cache
from typing import Any

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
from adaptive_shift_bench.scenarios import DEFAULT_TIMEOUT_S, TRANSFER_ONLY_MAX_TURNS


STRICT_BENCHMARK_SUITE = "v3_learning_strict"
STRICT_DATASET_VERSION = "3.0-strict-prior-proof"
STRICT_SURFACES = ("abstract", "realistic")
STRICT_DIFFICULTY_TIERS = ("standard", "hard")
STRICT_SEEDS = (1, 2)


def _doc(doc_id: str, title: str, text: str, *keywords: str) -> Document:
    return Document(doc_id=doc_id, title=title, text=text.strip(), keywords=keywords)


def _identifier(base: str, seed: int) -> str:
    return f"{base}_{seed}"


def _string_token(base: str, seed: int) -> str:
    return f"{base}-{seed}"


def _surface_language(surface_style: str) -> LanguageSurface:
    del surface_style
    return LanguageSurface.ENGLISH


def _surface_prompt_intro(package_name: str, task_target: str, regime: EvidenceRegime) -> str:
    if regime is EvidenceRegime.DOCS_SEARCH:
        return (
            f"You are solving a strict learning benchmark for {package_name}.\n"
            "The correct answer depends on local workspace evidence, not prior knowledge.\n"
            f"{task_target}\n"
            "Use the local search_docs and read_doc actions to inspect the workspace docs."
        )
    return (
        f"You are solving a strict learning benchmark for {package_name}.\n"
        "The correct answer depends on local workspace evidence, not prior knowledge.\n"
        f"{task_target}\n"
        "No local docs are available in this stage. Use only what you learned earlier in this workspace."
    )


def _build_strict_scenario(
    *,
    scenario_id: str,
    sequence_id: str,
    family: ScenarioFamily,
    package_name: str,
    task_target: str,
    docs: tuple[Document, ...],
    validator_kind: ValidatorKind,
    validator_payload: dict[str, Any],
    sequence_stage: SequenceStage,
    evidence_regime: EvidenceRegime,
    latent_rule_id: str,
    surface_style: str,
    difficulty_tier: str,
    forbidden_tokens: tuple[str, ...] = (),
    must_use_doc_ids: tuple[str, ...] = (),
    max_turns: int = 6,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        family=family,
        difficulty=difficulty_tier,
        evidence_regime=evidence_regime,
        prompt=_surface_prompt_intro(package_name, task_target, evidence_regime),
        docs_index=docs,
        tool_fixtures={"world_id": latent_rule_id},
        validator_spec=ValidatorSpec(
            kind=validator_kind,
            forbidden_tokens=forbidden_tokens,
            must_use_doc_ids=must_use_doc_ids,
            payload=validator_payload,
        ),
        max_turns=max_turns,
        timeout_s=DEFAULT_TIMEOUT_S,
        attempts=1,
        benchmark_suite=STRICT_BENCHMARK_SUITE,
        sequence_id=sequence_id,
        sequence_stage=sequence_stage,
        language_surface=_surface_language(surface_style),
        surface_style=surface_style,
        difficulty_tier=difficulty_tier,
        latent_rule_id=latent_rule_id,
    )


def _strict_python_payload(
    *,
    root_name: str,
    class_name: str | None,
    expected_path: str,
    expected_args: list[Any] | None,
    expected_kwargs: dict[str, Any],
    forbidden_paths: tuple[str, ...],
    scalars: tuple[str, ...],
) -> dict[str, Any]:
    root_operations = {
        root_name: [
            {"path": expected_path.removeprefix(f"{root_name}."), "behavior": "return"},
            *[
                {
                    "path": forbidden_path.removeprefix(f"{root_name}."),
                    "behavior": "deprecated",
                    "message": f"{forbidden_path} is deprecated in this local workspace",
                }
                for forbidden_path in forbidden_paths
            ],
        ]
    }
    mock_config: dict[str, Any] = {
        "root_operations": root_operations,
        "scalars": scalars,
    }
    if class_name is not None:
        mock_config["exported_classes"] = {class_name: root_name}
    payload: dict[str, Any] = {
        "mock": "strict_python",
        "mock_config": mock_config,
        "expected_call": {
            "path": expected_path,
            "kwargs": expected_kwargs,
        },
        "forbidden_call_paths": forbidden_paths,
    }
    if expected_args is not None:
        payload["expected_call"]["args"] = expected_args
    return payload


def _api_names(template_index: int, seed: int, surface_style: str) -> dict[str, Any]:
    if surface_style == "abstract":
        bases = {
            "package": ("Veloxa Grid", "Neru Relay", "Tavik Matrix", "Solen Arc"),
            "class": ("VeloxaClient", "NeruClient", "TavikClient", "SolenClient"),
            "legacy_path": (
                ("vekta", "rume"),
                ("soral", "minta"),
                ("tarin", "velor"),
                ("ulnar", "ketha"),
            ),
            "new_path": (
                ("zypha", "qorin"),
                ("nemra", "talis"),
                ("vorin", "pelta"),
                ("cirra", "muxen"),
            ),
            "alt_path": (
                ("zypha", "telu"),
                ("nemra", "qel"),
                ("vorin", "sanu"),
                ("cirra", "pavo"),
            ),
            "legacy_field": ("ivor", "pel", "rax", "sen"),
            "new_field": ("quill", "mora", "tide", "vanta"),
            "alt_field": ("quai", "mira", "tava", "venu"),
            "alt_model_field": ("engi", "dyna", "rota", "nive"),
        }
    else:
        bases = {
            "package": ("RelayForge SDK", "LanternHub SDK", "NimbusFlow SDK", "TransitMesh SDK"),
            "class": ("RelayForgeClient", "LanternHubClient", "NimbusFlowClient", "TransitMeshClient"),
            "legacy_path": (
                ("responses", "create"),
                ("dispatch", "create"),
                ("gateway", "submit"),
                ("router", "send"),
            ),
            "new_path": (
                ("relay", "submit"),
                ("dispatch", "deliver"),
                ("gateway", "publish"),
                ("router", "forward"),
            ),
            "alt_path": (
                ("bridge", "submit"),
                ("portal", "deliver"),
                ("switch", "publish"),
                ("mesh", "forward"),
            ),
            "legacy_field": ("input_text", "body_text", "payload_text", "message_text"),
            "new_field": ("packet_body", "request_body", "signal_body", "cargo_body"),
            "alt_field": ("bridge_body", "portal_body", "switch_body", "mesh_body"),
            "alt_model_field": ("engine", "engine", "channel", "channel"),
        }
    return {
        "package_name": bases["package"][template_index],
        "class_name": bases["class"][template_index],
        "legacy_path": ".".join(_identifier(part, seed) for part in bases["legacy_path"][template_index]),
        "new_path": ".".join(_identifier(part, seed) for part in bases["new_path"][template_index]),
        "alt_path": ".".join(_identifier(part, seed) for part in bases["alt_path"][template_index]),
        "legacy_field": _identifier(bases["legacy_field"][template_index], seed),
        "new_field": _identifier(bases["new_field"][template_index], seed),
        "alt_field": _identifier(bases["alt_field"][template_index], seed),
        "alt_model_field": _identifier(bases["alt_model_field"][template_index], seed),
        "model_name": _string_token(f"orbit-{template_index + 1}", seed),
        "adapt_text": _string_token(f"adapt-{template_index + 1}", seed),
        "transfer_text": _string_token(f"transfer-{template_index + 1}", seed),
        "capstone_text": _string_token(f"capstone-{template_index + 1}", seed),
        "root_name": "client",
        "alt_root_name": "bridge",
    }


def _dsl_names(template_index: int, seed: int, surface_style: str) -> dict[str, Any]:
    if surface_style == "abstract":
        bases = {
            "package": ("Kavon Tables", "Rydex Sheets", "Telor Frames", "Mivra Tables"),
            "method_old": ("meru_join", "selan_pick", "toren_sort", "vexa_filter"),
            "method_new": ("qera_stack", "pelan_keep", "sorin_rank", "nuvra_screen"),
            "method_alt": ("qera_weave", "pelan_slice", "sorin_order", "nuvra_gate"),
            "kw_old": ("nema_flag", "sila_keys", "tera_key", "vora_rule"),
            "kw_new": ("qeta_flag", "pila_cols", "sora_key", "nova_rule"),
            "kw_alt": ("qiri_flag", "pira_cols", "suni_key", "nira_rule"),
        }
    else:
        bases = {
            "package": ("FrameCraft", "SheetBridge", "OrderAxis", "FilterDock"),
            "method_old": ("concat_rows", "select_fields", "sort_rows", "filter_rows"),
            "method_new": ("stack_frames", "keep_fields", "rank_rows", "screen_rows"),
            "method_alt": ("weave_frames", "slice_fields", "order_rows", "gate_rows"),
            "kw_old": ("ignore_index", "names", "ascending", "condition"),
            "kw_new": ("reset_rows", "columns", "descending_key", "predicate"),
            "kw_alt": ("collapse_rows", "field_list", "priority_key", "gate_rule"),
        }
    task_tokens = (
        ("left", "right"),
        ("sheet",),
        ("sheet",),
        ("sheet",),
    )
    return {
        "package_name": bases["package"][template_index],
        "legacy_method": _identifier(bases["method_old"][template_index], seed),
        "new_method": _identifier(bases["method_new"][template_index], seed),
        "alt_method": _identifier(bases["method_alt"][template_index], seed),
        "legacy_kw": _identifier(bases["kw_old"][template_index], seed),
        "new_kw": _identifier(bases["kw_new"][template_index], seed),
        "alt_kw": _identifier(bases["kw_alt"][template_index], seed),
        "root_name": "tablekit",
        "alt_root_name": "grid_bridge",
        "scalars": task_tokens[template_index],
    }


def _registry_names(template_index: int, seed: int, surface_style: str) -> dict[str, Any]:
    if surface_style == "abstract":
        providers = ("Velor Registry", "Tarin Registry", "Qevra Registry", "Noran Registry")
        request_prefix = ("vek", "tar", "qev", "nor")
        fallback_prefix = ("vexa", "talis", "qyra", "nema")
        deprecated_prefix = ("vold", "tore", "qeld", "nira")
        field_prefix = ("moru", "tavin", "qiri", "nalen")
    else:
        providers = ("RelayMint Registry", "LanternCore Registry", "NimbusSpan Registry", "TransitDock Registry")
        request_prefix = ("relay", "lantern", "nimbus", "transit")
        fallback_prefix = ("bridge", "harbor", "vector", "reserve")
        deprecated_prefix = ("legacy", "fallback", "archive", "default")
        field_prefix = ("model", "target_model", "approved_model", "route_model")
    request_base = request_prefix[template_index]
    fallback_base = fallback_prefix[template_index]
    deprecated_base = deprecated_prefix[template_index]
    alt_field_base = field_prefix[template_index]
    return {
        "package_name": providers[template_index],
        "request_small": _string_token(f"{request_base}-small", seed),
        "request_transfer": _string_token(f"{request_base}-lite", seed),
        "request_capstone": _string_token(f"{request_base}-mini", seed),
        "fallback_model": _string_token(f"{fallback_base}-approved", seed),
        "deprecated_model": _string_token(f"{deprecated_base}-default", seed),
        "alt_field": _identifier(alt_field_base, seed),
    }


def _api_sequence_from_spec(
    template_index: int,
    template_id: str,
    seed: int,
    surface_style: str,
    difficulty_tier: str,
) -> ScenarioSequence:
    names = _api_names(template_index, seed, surface_style)
    latent_rule_id = f"{template_id}-seed-{seed}"
    sequence_id = f"v3-strict-{difficulty_tier}-{surface_style}-{template_id}-s{seed}"
    base_doc_prefix = f"{sequence_id}-doc"

    adapt_docs = [
        _doc(
            f"{base_doc_prefix}-route",
            "Local route bulletin",
            f"""
            The local request route changed in this workspace.
            Legacy route: client.{names['legacy_path']}(model=..., {names['legacy_field']}=...)
            Current route: client.{names['new_path']}(model=..., {names['new_field']}=...)
            """,
            "route",
            names["new_path"],
            names["new_field"],
        ),
        _doc(
            f"{base_doc_prefix}-policy",
            "Execution policy",
            f"""
            Only client.{names['new_path']} is accepted on the current route.
            Any use of client.{names['legacy_path']} should be treated as obsolete.
            """,
            "policy",
            names["new_path"],
            names["legacy_path"],
        ),
    ]
    adapt_required_docs = (adapt_docs[0].doc_id,)
    if difficulty_tier == "hard":
        adapt_docs.extend(
            [
                _doc(
                    f"{base_doc_prefix}-archived",
                    "Archived preview note",
                    f"""
                    Archived preview note:
                    client.{names['legacy_path']} still appears in historical notes.
                    Do not use historical preview paths for the current workspace.
                    """,
                    "archived",
                    names["legacy_path"],
                ),
                _doc(
                    f"{base_doc_prefix}-precedence",
                    "Doc precedence",
                    f"""
                    If multiple docs disagree, prefer the route bulletin and execution policy over archived notes.
                    The authoritative path is client.{names['new_path']}.
                    """,
                    "precedence",
                    names["new_path"],
                ),
            ]
        )
        adapt_required_docs = (adapt_docs[0].doc_id, adapt_docs[1].doc_id)

    cap_docs = [
        _doc(
            f"{base_doc_prefix}-capstone-map",
            "Alternate surface map",
            f"""
            Alternate local surface:
            client.{names['new_path']} -> bridge.{names['alt_path']}
            model -> {names['alt_model_field']}
            {names['new_field']} -> {names['alt_field']}
            """,
            "alternate",
            names["alt_path"],
            names["alt_field"],
            names["alt_model_field"],
        ),
        _doc(
            f"{base_doc_prefix}-capstone-policy",
            "Alternate surface policy",
            f"""
            Use bridge.{names['alt_path']} on the alternate surface.
            The direct client.{names['new_path']} path is not valid on the alternate surface.
            """,
            "alternate",
            names["alt_path"],
        ),
    ]
    cap_required_docs = (cap_docs[0].doc_id,)
    if difficulty_tier == "hard":
        cap_docs.extend(
            [
                _doc(
                    f"{base_doc_prefix}-capstone-distractor",
                    "Legacy bridge note",
                    f"""
                    Historical bridge docs may mention bridge.{names['new_path']}.
                    Those notes are obsolete for the alternate surface.
                    """,
                    "legacy",
                    names["new_path"],
                ),
                _doc(
                    f"{base_doc_prefix}-capstone-precedence",
                    "Bridge precedence",
                    f"""
                    Prefer the alternate surface map and alternate surface policy over historical bridge notes.
                    """,
                    "precedence",
                    names["alt_path"],
                ),
            ]
        )
        cap_required_docs = (cap_docs[0].doc_id, cap_docs[1].doc_id)

    prior_task = (
        f"Assume `client = {names['class_name']}()` already exists. Return only Python code that sends "
        f"`{names['adapt_text']}` to model `{names['model_name']}` using the newest local request route."
    )
    transfer_task = (
        f"Assume `client = {names['class_name']}()` already exists. Return only Python code that sends "
        f"`{names['transfer_text']}` to model `{names['model_name']}` using the newest local request route."
    )
    capstone_task = (
        f"Assume `bridge` already exists. Return only Python code that sends "
        f"`{names['capstone_text']}` to `{names['model_name']}` through the alternate local surface."
    )
    scalars = ()
    prior_payload = _strict_python_payload(
        root_name=names["root_name"],
        class_name=names["class_name"],
        expected_path=f"{names['root_name']}.{names['new_path']}",
        expected_args=None,
        expected_kwargs={"model": names["model_name"], names["new_field"]: names["adapt_text"]},
        forbidden_paths=(f"{names['root_name']}.{names['legacy_path']}",),
        scalars=scalars,
    )
    cap_payload = _strict_python_payload(
        root_name=names["alt_root_name"],
        class_name=None,
        expected_path=f"{names['alt_root_name']}.{names['alt_path']}",
        expected_args=None,
        expected_kwargs={
            names["alt_model_field"]: names["model_name"],
            names["alt_field"]: names["capstone_text"],
        },
        forbidden_paths=(f"{names['alt_root_name']}.{names['new_path']}",),
        scalars=scalars,
    )
    stages = (
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-prior-probe",
            sequence_id=sequence_id,
            family=ScenarioFamily.API_MIGRATION,
            package_name=names["package_name"],
            task_target=prior_task,
            docs=(),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=prior_payload,
            sequence_stage=SequenceStage.TEACH,
            evidence_regime=EvidenceRegime.TRANSFER_ONLY,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["legacy_path"],),
            max_turns=TRANSFER_ONLY_MAX_TURNS,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-adapt",
            sequence_id=sequence_id,
            family=ScenarioFamily.API_MIGRATION,
            package_name=names["package_name"],
            task_target=prior_task,
            docs=tuple(adapt_docs),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=prior_payload,
            sequence_stage=SequenceStage.ADAPT,
            evidence_regime=EvidenceRegime.DOCS_SEARCH,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["legacy_path"],),
            must_use_doc_ids=adapt_required_docs,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-transfer",
            sequence_id=sequence_id,
            family=ScenarioFamily.API_MIGRATION,
            package_name=names["package_name"],
            task_target=transfer_task,
            docs=(),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=_strict_python_payload(
                root_name=names["root_name"],
                class_name=names["class_name"],
                expected_path=f"{names['root_name']}.{names['new_path']}",
                expected_args=None,
                expected_kwargs={"model": names["model_name"], names["new_field"]: names["transfer_text"]},
                forbidden_paths=(f"{names['root_name']}.{names['legacy_path']}",),
                scalars=scalars,
            ),
            sequence_stage=SequenceStage.TRANSFER,
            evidence_regime=EvidenceRegime.TRANSFER_ONLY,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["legacy_path"],),
            max_turns=TRANSFER_ONLY_MAX_TURNS,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-capstone",
            sequence_id=sequence_id,
            family=ScenarioFamily.API_MIGRATION,
            package_name=names["package_name"],
            task_target=capstone_task,
            docs=tuple(cap_docs),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=cap_payload,
            sequence_stage=SequenceStage.CAPSTONE,
            evidence_regime=EvidenceRegime.DOCS_SEARCH,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["new_path"],),
            must_use_doc_ids=cap_required_docs,
        ),
    )
    return ScenarioSequence(
        id=sequence_id,
        family=ScenarioFamily.API_MIGRATION,
        stages=stages,
        benchmark_suite=STRICT_BENCHMARK_SUITE,
        latent_rule_id=latent_rule_id,
        surface_style=surface_style,
        difficulty_tier=difficulty_tier,
    )


def _dsl_expected_parts(template_index: int, names: dict[str, Any], stage: str) -> tuple[str, list[Any], dict[str, Any], tuple[str, ...]]:
    if template_index == 0:
        value_kw = names["new_kw"] if stage != "capstone" else names["alt_kw"]
        method = names["new_method"] if stage != "capstone" else names["alt_method"]
        root = names["root_name"] if stage != "capstone" else names["alt_root_name"]
        forbidden = (f"{root}.{names['legacy_method']}",)
        return f"{root}.{method}", [["left", "right"]], {value_kw: True}, forbidden
    if template_index == 1:
        value_kw = names["new_kw"] if stage != "capstone" else names["alt_kw"]
        method = names["new_method"] if stage != "capstone" else names["alt_method"]
        root = names["root_name"] if stage != "capstone" else names["alt_root_name"]
        target_columns = ["city", "margin"] if stage == "transfer" else ["city", "revenue"]
        forbidden = (f"{root}.{names['legacy_method']}",)
        return f"{root}.{method}", ["sheet"], {value_kw: target_columns}, forbidden
    if template_index == 2:
        value_kw = names["new_kw"] if stage != "capstone" else names["alt_kw"]
        method = names["new_method"] if stage != "capstone" else names["alt_method"]
        root = names["root_name"] if stage != "capstone" else names["alt_root_name"]
        target_key = "margin" if stage == "transfer" else "revenue"
        forbidden = (f"{root}.{names['legacy_method']}",)
        return f"{root}.{method}", ["sheet"], {value_kw: target_key}, forbidden
    value_kw = names["new_kw"] if stage != "capstone" else names["alt_kw"]
    method = names["new_method"] if stage != "capstone" else names["alt_method"]
    root = names["root_name"] if stage != "capstone" else names["alt_root_name"]
    target_rule = "margin > 0" if stage == "transfer" else "revenue > 0"
    forbidden = (f"{root}.{names['legacy_method']}",)
    return f"{root}.{method}", ["sheet"], {value_kw: target_rule}, forbidden


def _dsl_sequence_from_spec(
    template_index: int,
    template_id: str,
    seed: int,
    surface_style: str,
    difficulty_tier: str,
) -> ScenarioSequence:
    names = _dsl_names(template_index, seed, surface_style)
    latent_rule_id = f"{template_id}-seed-{seed}"
    sequence_id = f"v3-strict-{difficulty_tier}-{surface_style}-{template_id}-s{seed}"
    base_doc_prefix = f"{sequence_id}-doc"

    adapt_docs = [
        _doc(
            f"{base_doc_prefix}-mapping",
            "Compatibility mapping",
            f"""
            Legacy operation: {names['root_name']}.{names['legacy_method']}
            Current operation: {names['root_name']}.{names['new_method']}
            Legacy keyword: {names['legacy_kw']}
            Current keyword: {names['new_kw']}
            """,
            names["new_method"],
            names["new_kw"],
        ),
        _doc(
            f"{base_doc_prefix}-policy",
            "Compatibility policy",
            f"""
            Use {names['root_name']}.{names['new_method']} with {names['new_kw']}.
            Do not use {names['root_name']}.{names['legacy_method']} in the current workspace.
            """,
            names["new_method"],
            names["legacy_method"],
        ),
    ]
    adapt_required_docs = (adapt_docs[0].doc_id,)
    if difficulty_tier == "hard":
        adapt_docs.extend(
            [
                _doc(
                    f"{base_doc_prefix}-archive",
                    "Archive note",
                    f"""
                    Historical notebooks may still show {names['root_name']}.{names['legacy_method']}.
                    Those historical notebooks are obsolete in the current workspace.
                    """,
                    names["legacy_method"],
                ),
                _doc(
                    f"{base_doc_prefix}-precedence",
                    "Mapping precedence",
                    f"""
                    Prefer the compatibility mapping and compatibility policy over historical notebooks.
                    """,
                    names["new_method"],
                ),
            ]
        )
        adapt_required_docs = (adapt_docs[0].doc_id, adapt_docs[1].doc_id)

    cap_docs = [
        _doc(
            f"{base_doc_prefix}-alt-map",
            "Alternate wrapper map",
            f"""
            Alternate wrapper:
            {names['root_name']}.{names['new_method']} -> {names['alt_root_name']}.{names['alt_method']}
            {names['new_kw']} -> {names['alt_kw']}
            """,
            names["alt_method"],
            names["alt_kw"],
        ),
        _doc(
            f"{base_doc_prefix}-alt-policy",
            "Alternate wrapper policy",
            f"""
            Use {names['alt_root_name']}.{names['alt_method']} on the alternate wrapper.
            """,
            names["alt_method"],
        ),
    ]
    cap_required_docs = (cap_docs[0].doc_id,)
    if difficulty_tier == "hard":
        cap_docs.extend(
            [
                _doc(
                    f"{base_doc_prefix}-alt-archive",
                    "Legacy alternate note",
                    f"""
                    Historical wrapper notes may mention {names['alt_root_name']}.{names['new_method']}.
                    Those notes are obsolete.
                    """,
                    names["new_method"],
                ),
                _doc(
                    f"{base_doc_prefix}-alt-precedence",
                    "Alternate wrapper precedence",
                    f"""
                    Prefer the alternate wrapper map and alternate wrapper policy over historical wrapper notes.
                    """,
                    names["alt_method"],
                ),
            ]
        )
        cap_required_docs = (cap_docs[0].doc_id, cap_docs[1].doc_id)

    if template_index == 0:
        task_text = (
            f"Assume `tablekit`, `left`, and `right` already exist. Return only Python code that combines "
            "`left` and `right` with the newest local operation while resetting row ids."
        )
        transfer_task = task_text
        capstone_task = (
            "Assume `grid_bridge`, `left`, and `right` already exist. Return only Python code that performs "
            "the same combine operation on the alternate wrapper."
        )
    elif template_index == 1:
        task_text = (
            "Assume `tablekit` and `sheet` already exist. Return only Python code that keeps columns "
            "`['city', 'revenue']` using the newest local selector."
        )
        transfer_task = (
            "Assume `tablekit` and `sheet` already exist. Return only Python code that keeps columns "
            "`['city', 'margin']` using the newest local selector."
        )
        capstone_task = (
            "Assume `grid_bridge` and `sheet` already exist. Return only Python code that keeps columns "
            "`['city', 'revenue']` using the alternate wrapper."
        )
    elif template_index == 2:
        task_text = (
            "Assume `tablekit` and `sheet` already exist. Return only Python code that orders `sheet` by "
            "`revenue` using the newest local sorter."
        )
        transfer_task = (
            "Assume `tablekit` and `sheet` already exist. Return only Python code that orders `sheet` by "
            "`margin` using the newest local sorter."
        )
        capstone_task = (
            "Assume `grid_bridge` and `sheet` already exist. Return only Python code that orders `sheet` by "
            "`revenue` using the alternate wrapper."
        )
    else:
        task_text = (
            "Assume `tablekit` and `sheet` already exist. Return only Python code that keeps rows where "
            "`revenue > 0` using the newest local filter."
        )
        transfer_task = (
            "Assume `tablekit` and `sheet` already exist. Return only Python code that keeps rows where "
            "`margin > 0` using the newest local filter."
        )
        capstone_task = (
            "Assume `grid_bridge` and `sheet` already exist. Return only Python code that keeps rows where "
            "`revenue > 0` using the alternate wrapper."
        )

    def _payload(stage: str) -> dict[str, Any]:
        expected_path, expected_args, expected_kwargs, forbidden = _dsl_expected_parts(template_index, names, stage)
        root = expected_path.split(".", 1)[0]
        return _strict_python_payload(
            root_name=root,
            class_name=None,
            expected_path=expected_path,
            expected_args=expected_args,
            expected_kwargs=expected_kwargs,
            forbidden_paths=forbidden,
            scalars=names["scalars"],
        )

    stages = (
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-prior-probe",
            sequence_id=sequence_id,
            family=ScenarioFamily.DSL_WRAPPER,
            package_name=names["package_name"],
            task_target=task_text,
            docs=(),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=_payload("adapt"),
            sequence_stage=SequenceStage.TEACH,
            evidence_regime=EvidenceRegime.TRANSFER_ONLY,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-adapt",
            sequence_id=sequence_id,
            family=ScenarioFamily.DSL_WRAPPER,
            package_name=names["package_name"],
            task_target=task_text,
            docs=tuple(adapt_docs),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=_payload("adapt"),
            sequence_stage=SequenceStage.ADAPT,
            evidence_regime=EvidenceRegime.DOCS_SEARCH,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            must_use_doc_ids=adapt_required_docs,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-transfer",
            sequence_id=sequence_id,
            family=ScenarioFamily.DSL_WRAPPER,
            package_name=names["package_name"],
            task_target=transfer_task,
            docs=(),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=_payload("transfer"),
            sequence_stage=SequenceStage.TRANSFER,
            evidence_regime=EvidenceRegime.TRANSFER_ONLY,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            max_turns=TRANSFER_ONLY_MAX_TURNS,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-capstone",
            sequence_id=sequence_id,
            family=ScenarioFamily.DSL_WRAPPER,
            package_name=names["package_name"],
            task_target=capstone_task,
            docs=tuple(cap_docs),
            validator_kind=ValidatorKind.PYTHON_MOCK,
            validator_payload=_payload("capstone"),
            sequence_stage=SequenceStage.CAPSTONE,
            evidence_regime=EvidenceRegime.DOCS_SEARCH,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            must_use_doc_ids=cap_required_docs,
        ),
    )
    return ScenarioSequence(
        id=sequence_id,
        family=ScenarioFamily.DSL_WRAPPER,
        stages=stages,
        benchmark_suite=STRICT_BENCHMARK_SUITE,
        latent_rule_id=latent_rule_id,
        surface_style=surface_style,
        difficulty_tier=difficulty_tier,
    )


def _registry_sequence_from_spec(
    template_index: int,
    template_id: str,
    seed: int,
    surface_style: str,
    difficulty_tier: str,
) -> ScenarioSequence:
    names = _registry_names(template_index, seed, surface_style)
    latent_rule_id = f"{template_id}-seed-{seed}"
    sequence_id = f"v3-strict-{difficulty_tier}-{surface_style}-{template_id}-s{seed}"
    base_doc_prefix = f"{sequence_id}-doc"

    adapt_docs = [
        _doc(
            f"{base_doc_prefix}-table",
            "Fallback table",
            f"""
            {names['request_small']} is not registered locally.
            Use {names['fallback_model']} as the sanctioned local fallback for this request class.
            """,
            names["request_small"],
            names["fallback_model"],
        ),
        _doc(
            f"{base_doc_prefix}-policy",
            "Registry policy",
            f"""
            Prefer the sanctioned fallback {names['fallback_model']} over deprecated defaults such as {names['deprecated_model']}.
            """,
            names["fallback_model"],
            names["deprecated_model"],
        ),
    ]
    adapt_required_docs = (adapt_docs[0].doc_id,)
    if difficulty_tier == "hard":
        adapt_docs.extend(
            [
                _doc(
                    f"{base_doc_prefix}-archive",
                    "Archived fallback note",
                    f"""
                    Archived notes may still mention {names['deprecated_model']}.
                    Do not use archived defaults for the current route.
                    """,
                    names["deprecated_model"],
                ),
                _doc(
                    f"{base_doc_prefix}-precedence",
                    "Registry precedence",
                    f"""
                    Prefer the fallback table and registry policy over archived notes.
                    """,
                    names["fallback_model"],
                ),
            ]
        )
        adapt_required_docs = (adapt_docs[0].doc_id, adapt_docs[1].doc_id)

    cap_docs = [
        _doc(
            f"{base_doc_prefix}-cap-map",
            "Alternate schema map",
            f"""
            On the alternate selector, the JSON field name changes from `model` to `{names['alt_field']}`.
            The sanctioned fallback remains {names['fallback_model']}.
            """,
            names["alt_field"],
            names["fallback_model"],
        ),
        _doc(
            f"{base_doc_prefix}-cap-policy",
            "Alternate schema policy",
            f"""
            Use exactly one JSON field named `{names['alt_field']}` on the alternate selector.
            """,
            names["alt_field"],
        ),
    ]
    cap_required_docs = (cap_docs[0].doc_id,)
    if difficulty_tier == "hard":
        cap_docs.extend(
            [
                _doc(
                    f"{base_doc_prefix}-cap-archive",
                    "Legacy selector note",
                    f"""
                    Historical selectors may still emit `model`.
                    Do not use historical selector fields on the alternate selector.
                    """,
                    "model",
                ),
                _doc(
                    f"{base_doc_prefix}-cap-precedence",
                    "Alternate schema precedence",
                    f"""
                    Prefer the alternate schema map and alternate schema policy over historical selector notes.
                    """,
                    names["alt_field"],
                ),
            ]
        )
        cap_required_docs = (cap_docs[0].doc_id, cap_docs[1].doc_id)

    base_prompt = (
        f"Return only a JSON object with a single `model` field using the sanctioned fallback for request "
        f"class `{names['request_small']}`."
    )
    transfer_prompt = (
        f"Return only a JSON object with a single `model` field using the sanctioned fallback for request "
        f"class `{names['request_transfer']}`."
    )
    capstone_prompt = (
        f"Return only a JSON object with a single `{names['alt_field']}` field using the sanctioned fallback for "
        f"request class `{names['request_capstone']}` on the alternate selector."
    )
    stages = (
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-prior-probe",
            sequence_id=sequence_id,
            family=ScenarioFamily.FUTURE_REGISTRY,
            package_name=names["package_name"],
            task_target=base_prompt,
            docs=(),
            validator_kind=ValidatorKind.MODEL_REGISTRY,
            validator_payload={
                "registry": {
                    names["request_small"]: {"available": False, "compatible": False},
                    names["request_transfer"]: {"available": False, "compatible": False},
                    names["request_capstone"]: {"available": False, "compatible": False},
                    names["fallback_model"]: {"available": True, "compatible": True},
                    names["deprecated_model"]: {"available": True, "compatible": True},
                },
                "expected_model": names["fallback_model"],
                "schema_fields": ("model",),
            },
            sequence_stage=SequenceStage.TEACH,
            evidence_regime=EvidenceRegime.TRANSFER_ONLY,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["deprecated_model"],),
            max_turns=TRANSFER_ONLY_MAX_TURNS,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-adapt",
            sequence_id=sequence_id,
            family=ScenarioFamily.FUTURE_REGISTRY,
            package_name=names["package_name"],
            task_target=base_prompt,
            docs=tuple(adapt_docs),
            validator_kind=ValidatorKind.MODEL_REGISTRY,
            validator_payload={
                "registry": {
                    names["request_small"]: {"available": False, "compatible": False},
                    names["request_transfer"]: {"available": False, "compatible": False},
                    names["request_capstone"]: {"available": False, "compatible": False},
                    names["fallback_model"]: {"available": True, "compatible": True},
                    names["deprecated_model"]: {"available": True, "compatible": True},
                },
                "expected_model": names["fallback_model"],
                "schema_fields": ("model",),
            },
            sequence_stage=SequenceStage.ADAPT,
            evidence_regime=EvidenceRegime.DOCS_SEARCH,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["deprecated_model"],),
            must_use_doc_ids=adapt_required_docs,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-transfer",
            sequence_id=sequence_id,
            family=ScenarioFamily.FUTURE_REGISTRY,
            package_name=names["package_name"],
            task_target=transfer_prompt,
            docs=(),
            validator_kind=ValidatorKind.MODEL_REGISTRY,
            validator_payload={
                "registry": {
                    names["request_small"]: {"available": False, "compatible": False},
                    names["request_transfer"]: {"available": False, "compatible": False},
                    names["request_capstone"]: {"available": False, "compatible": False},
                    names["fallback_model"]: {"available": True, "compatible": True},
                    names["deprecated_model"]: {"available": True, "compatible": True},
                },
                "expected_model": names["fallback_model"],
                "schema_fields": ("model",),
            },
            sequence_stage=SequenceStage.TRANSFER,
            evidence_regime=EvidenceRegime.TRANSFER_ONLY,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["deprecated_model"],),
            max_turns=TRANSFER_ONLY_MAX_TURNS,
        ),
        _build_strict_scenario(
            scenario_id=f"{sequence_id}-capstone",
            sequence_id=sequence_id,
            family=ScenarioFamily.FUTURE_REGISTRY,
            package_name=names["package_name"],
            task_target=capstone_prompt,
            docs=tuple(cap_docs),
            validator_kind=ValidatorKind.MODEL_REGISTRY,
            validator_payload={
                "registry": {
                    names["request_small"]: {"available": False, "compatible": False},
                    names["request_transfer"]: {"available": False, "compatible": False},
                    names["request_capstone"]: {"available": False, "compatible": False},
                    names["fallback_model"]: {"available": True, "compatible": True},
                    names["deprecated_model"]: {"available": True, "compatible": True},
                },
                "expected_model": names["fallback_model"],
                "schema_fields": (names["alt_field"],),
            },
            sequence_stage=SequenceStage.CAPSTONE,
            evidence_regime=EvidenceRegime.DOCS_SEARCH,
            latent_rule_id=latent_rule_id,
            surface_style=surface_style,
            difficulty_tier=difficulty_tier,
            forbidden_tokens=(names["deprecated_model"],),
            must_use_doc_ids=cap_required_docs,
        ),
    )
    return ScenarioSequence(
        id=sequence_id,
        family=ScenarioFamily.FUTURE_REGISTRY,
        stages=stages,
        benchmark_suite=STRICT_BENCHMARK_SUITE,
        latent_rule_id=latent_rule_id,
        surface_style=surface_style,
        difficulty_tier=difficulty_tier,
    )


_STRICT_API_TEMPLATE_IDS = (
    "api-route-payload",
    "api-entrypoint-shift",
    "api-request-body",
    "api-route-forward",
)
_STRICT_DSL_TEMPLATE_IDS = (
    "dsl-stack-rows",
    "dsl-keep-columns",
    "dsl-rank-rows",
    "dsl-screen-rows",
)
_STRICT_REGISTRY_TEMPLATE_IDS = (
    "registry-small-fallback",
    "registry-compact-fallback",
    "registry-budget-fallback",
    "registry-route-fallback",
)


@lru_cache(maxsize=1)
def build_v3_learning_strict_sequences(
    surface_style: str | None = None,
    difficulty_tier: str | None = None,
) -> tuple[ScenarioSequence, ...]:
    sequences: list[ScenarioSequence] = []
    surfaces = (surface_style,) if surface_style is not None else STRICT_SURFACES
    tiers = (difficulty_tier,) if difficulty_tier is not None else STRICT_DIFFICULTY_TIERS
    for tier in tiers:
        for surface in surfaces:
            for template_index, template_id in enumerate(_STRICT_API_TEMPLATE_IDS):
                for seed in STRICT_SEEDS:
                    sequences.append(_api_sequence_from_spec(template_index, template_id, seed, surface, tier))
            for template_index, template_id in enumerate(_STRICT_DSL_TEMPLATE_IDS):
                for seed in STRICT_SEEDS:
                    sequences.append(_dsl_sequence_from_spec(template_index, template_id, seed, surface, tier))
            for template_index, template_id in enumerate(_STRICT_REGISTRY_TEMPLATE_IDS):
                for seed in STRICT_SEEDS:
                    sequences.append(_registry_sequence_from_spec(template_index, template_id, seed, surface, tier))
    return tuple(sequences)


@lru_cache(maxsize=1)
def build_v3_learning_strict_stage_suite(
    surface_style: str | None = None,
    difficulty_tier: str | None = None,
) -> tuple[Scenario, ...]:
    return tuple(
        stage
        for sequence in build_v3_learning_strict_sequences(surface_style=surface_style, difficulty_tier=difficulty_tier)
        for stage in sequence.stages
    )


@lru_cache(maxsize=1)
def _v3_learning_strict_sequence_index() -> dict[str, ScenarioSequence]:
    return {sequence.id: sequence for sequence in build_v3_learning_strict_sequences()}


@lru_cache(maxsize=1)
def _v3_learning_strict_scenario_index() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in build_v3_learning_strict_stage_suite()}


def get_v3_learning_strict_sequence(sequence_id: str) -> ScenarioSequence:
    try:
        return _v3_learning_strict_sequence_index()[sequence_id]
    except KeyError as exc:
        raise KeyError(f"Unknown strict learning sequence id: {sequence_id}") from exc


def get_v3_learning_strict_scenario(scenario_id: str) -> Scenario:
    try:
        return _v3_learning_strict_scenario_index()[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown strict learning scenario id: {scenario_id}") from exc
