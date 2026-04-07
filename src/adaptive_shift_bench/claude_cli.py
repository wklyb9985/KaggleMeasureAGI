from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_shift_bench.llm import LLMAdapter


@dataclass(frozen=True)
class ClaudeCLIInvocation:
    prompt: str
    response_text: str
    raw_payload: dict
    duration_ms: int
    total_cost_usd: float
    model_usage: dict
    session_id: str | None


def build_transcript_prompt(history: list[tuple[str, str]], message: str) -> str:
    lines = [
        "You are being called through Claude Code print mode inside a benchmark harness.",
        "Treat the transcript below as the full conversation state.",
        "Reply only with the next assistant message for the current turn.",
        "",
    ]
    for index, (user_message, assistant_message) in enumerate(history, start=1):
        lines.extend(
            [
                f"<turn index=\"{index}\">",
                "<user>",
                user_message,
                "</user>",
                "<assistant>",
                assistant_message,
                "</assistant>",
                "</turn>",
                "",
            ]
        )

    lines.extend(
        [
            "<current_turn>",
            "<user>",
            message,
            "</user>",
            "</current_turn>",
        ]
    )
    return "\n".join(lines)


@dataclass
class ClaudeCLIAdapter(LLMAdapter):
    model: str
    workspace_dir: str | Path
    max_budget_usd: float = 1.50
    effort: str = "medium"
    history: list[tuple[str, str]] = field(default_factory=list)
    invocations: list[ClaudeCLIInvocation] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()
        self.invocations.clear()

    def prompt(self, message: str) -> str:
        transcript_prompt = build_transcript_prompt(self.history, message)
        command = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--tools",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--max-budget-usd",
            f"{self.max_budget_usd:.2f}",
            "--effort",
            self.effort,
            transcript_prompt,
        ]
        completed = subprocess.run(
            command,
            cwd=Path(self.workspace_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(
                f"claude CLI failed for model={self.model} returncode={completed.returncode} "
                f"stdout={stdout!r} stderr={stderr!r}"
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude CLI did not return valid JSON: {completed.stdout!r}") from exc

        response_text = str(payload["result"])
        invocation = ClaudeCLIInvocation(
            prompt=message,
            response_text=response_text,
            raw_payload=payload,
            duration_ms=int(payload.get("duration_ms", 0)),
            total_cost_usd=float(payload.get("total_cost_usd", 0.0)),
            model_usage=payload.get("modelUsage", {}),
            session_id=payload.get("session_id"),
        )
        self.invocations.append(invocation)
        self.history.append((message, response_text))
        return response_text

