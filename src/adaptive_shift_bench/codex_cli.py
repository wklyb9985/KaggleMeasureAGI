from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_shift_bench.claude_cli import build_transcript_prompt
from adaptive_shift_bench.llm import LLMAdapter


@dataclass(frozen=True)
class CodexCLIInvocation:
    prompt: str
    response_text: str
    raw_events: tuple[dict, ...]
    duration_ms: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    thread_id: str | None


@dataclass
class CodexCLIAdapter(LLMAdapter):
    model: str
    workspace_dir: str | Path
    effort: str = "medium"
    history: list[tuple[str, str]] = field(default_factory=list)
    invocations: list[CodexCLIInvocation] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()
        self.invocations.clear()

    def prompt(self, message: str) -> str:
        transcript_prompt = build_transcript_prompt(self.history, message)
        command = [
            "codex",
            "-a",
            "never",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-C",
            str(self.workspace_dir),
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.effort}"',
            transcript_prompt,
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(
                f"codex CLI failed for model={self.model} returncode={completed.returncode} "
                f"stdout={stdout!r} stderr={stderr!r}"
            )

        events: list[dict] = []
        response_text: str | None = None
        thread_id: str | None = None
        usage: dict[str, int] = {}
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")
            elif event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    response_text = str(item.get("text", ""))
            elif event.get("type") == "turn.completed":
                usage = event.get("usage", {})

        if response_text is None:
            raise RuntimeError(f"codex CLI did not return an agent message: {completed.stdout!r}")

        invocation = CodexCLIInvocation(
            prompt=message,
            response_text=response_text,
            raw_events=tuple(events),
            duration_ms=duration_ms,
            input_tokens=int(usage.get("input_tokens", 0)),
            cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            thread_id=thread_id,
        )
        self.invocations.append(invocation)
        self.history.append((message, response_text))
        return response_text
