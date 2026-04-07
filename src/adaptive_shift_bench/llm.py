from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class LLMAdapter(Protocol):
    def reset(self) -> None:
        ...

    def prompt(self, message: str) -> str:
        ...


@dataclass
class CallableAdapter:
    callback: Callable[[str], str]

    def reset(self) -> None:
        return None

    def prompt(self, message: str) -> str:
        return self.callback(message)


@dataclass
class ScriptedLLMAdapter:
    responses: list[str]
    cursor: int = 0
    seen_prompts: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.cursor = 0
        self.seen_prompts.clear()

    def prompt(self, message: str) -> str:
        self.seen_prompts.append(message)
        if self.cursor >= len(self.responses):
            raise RuntimeError("ScriptedLLMAdapter ran out of scripted responses.")
        response = self.responses[self.cursor]
        self.cursor += 1
        return response

