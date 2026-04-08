from __future__ import annotations

import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable


_STATE = threading.local()


def _session_stack() -> list[str]:
    stack = getattr(_STATE, "session_stack", None)
    if stack is None:
        stack = []
        _STATE.session_stack = stack
    return stack


@contextmanager
def _activate_session(session_key: str):
    stack = _session_stack()
    stack.append(session_key)
    try:
        yield
    finally:
        stack.pop()


def current_session_key() -> str | None:
    stack = _session_stack()
    if not stack:
        return None
    return stack[-1]


class _ChatContext:
    def __init__(self, chat_name: str, *, orphan: bool):
        suffix = uuid.uuid4().hex[:8] if orphan else "shared"
        self.session_key = f"chat:{chat_name}:{suffix}"

    def __enter__(self) -> str:
        self._manager = _activate_session(self.session_key)
        self._manager.__enter__()
        return self.session_key

    def __exit__(self, exc_type, exc, tb) -> None:
        self._manager.__exit__(exc_type, exc, tb)
        return None


class LocalChats:
    def new(self, chat_name: str, orphan: bool = False) -> _ChatContext:
        return _ChatContext(chat_name, orphan=orphan)


class LocalTask:
    def __init__(self, func: Callable[..., Any], name: str):
        self.func = func
        self.name = name
        self.__name__ = getattr(func, "__name__", name)
        self.__doc__ = getattr(func, "__doc__", None)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def run(self, *args, **kwargs):
        session_key = f"task:{self.name}:{uuid.uuid4().hex[:8]}"
        with _activate_session(session_key):
            return self.func(*args, **kwargs)


def task(*, name: str) -> Callable[[Callable[..., Any]], LocalTask]:
    def decorator(func: Callable[..., Any]) -> LocalTask:
        return LocalTask(func, name)

    return decorator


def run_parallel(cases: Iterable[Any], max_workers: int = 4) -> list[Any]:
    normalized: list[Callable[[], Any]] = []
    for case in cases:
        if callable(case):
            normalized.append(case)
            continue
        if isinstance(case, tuple) and len(case) == 2:
            task_obj, kwargs = case
            normalized.append(lambda task_obj=task_obj, kwargs=kwargs: task_obj.run(**kwargs))
            continue
        if isinstance(case, dict) and "task" in case:
            task_obj = case["task"]
            kwargs = case.get("kwargs", {})
            normalized.append(lambda task_obj=task_obj, kwargs=kwargs: task_obj.run(**kwargs))
            continue
        raise TypeError("run_parallel cases must be callables, (task, kwargs) tuples, or {'task': ..., 'kwargs': ...} dicts.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(case) for case in normalized]
        return [future.result() for future in futures]


@dataclass
class LocalTaskLLM:
    adapter_factory: Callable[[str], Any]
    adapters: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def prompt(self, message: str) -> str:
        session_key = current_session_key() or f"adhoc:{threading.get_ident()}:{uuid.uuid4().hex[:8]}"
        with self._lock:
            adapter = self.adapters.get(session_key)
            if adapter is None:
                adapter = self.adapter_factory(session_key)
                self.adapters[session_key] = adapter
        return adapter.prompt(message)

    def reset(self) -> None:
        with self._lock:
            for adapter in self.adapters.values():
                reset = getattr(adapter, "reset", None)
                if callable(reset):
                    reset()


def _build_mock_module() -> ModuleType:
    module = ModuleType("kaggle_benchmarks")
    module.task = task
    module.chats = LocalChats()
    module.run_parallel = run_parallel
    module.__dict__["_adaptive_shift_local_mock"] = True
    return module


def install_local_kaggle_benchmarks(*, force: bool = False) -> ModuleType:
    existing = sys.modules.get("kaggle_benchmarks")
    if existing is not None and not force:
        return existing
    module = _build_mock_module()
    sys.modules["kaggle_benchmarks"] = module
    return module


@contextmanager
def patched_local_kaggle_benchmarks(*, force: bool = True):
    previous = sys.modules.get("kaggle_benchmarks")
    module = install_local_kaggle_benchmarks(force=force)
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop("kaggle_benchmarks", None)
        else:
            sys.modules["kaggle_benchmarks"] = previous
