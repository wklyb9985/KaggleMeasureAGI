from __future__ import annotations

import json
import re

from adaptive_shift_bench.models import EpisodeAction


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_block(text: str) -> str | None:
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_action(text: str) -> EpisodeAction:
    payload_text = _extract_json_block(text)
    if payload_text is None:
        return EpisodeAction(action="answer", content=text.strip(), raw_response=text)

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return EpisodeAction(action="answer", content=text.strip(), raw_response=text)

    if not isinstance(payload, dict) or "action" not in payload:
        return EpisodeAction(action="answer", content=payload, raw_response=text)

    action = str(payload.get("action", "answer")).strip().lower()
    aliases = {
        "search": "search_docs",
        "search_local_docs": "search_docs",
        "read": "read_doc",
        "read_local_doc": "read_doc",
        "open_doc": "read_doc",
        "run": "run_candidate",
        "test": "run_candidate",
        "final": "answer",
    }
    normalized_action = aliases.get(action, action)
    return EpisodeAction(
        action=normalized_action,
        query=payload.get("query"),
        doc_id=payload.get("doc_id"),
        candidate=payload.get("candidate"),
        model=payload.get("model"),
        content=payload.get("content"),
        raw_response=text,
    )
