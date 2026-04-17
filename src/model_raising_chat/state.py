"""Module-level singletons. Imported by everything that needs the live state."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from .config import AppCfg, ModelRegistry


cfg: AppCfg = AppCfg.load()
registry: ModelRegistry = ModelRegistry()


# Cap audit log retention; long audits (max_turns=300, parallel subagents)
# would otherwise grow this unbounded.
AUDIT_LOG_MAX = 4000


@dataclass
class AuditState:
    """Per-model audit state held in-memory (tasks/queues never go in app.storage)."""
    task: Optional[asyncio.Task] = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=AUDIT_LOG_MAX))
    summary: Optional[dict] = None  # {"summary": str, "findings": [str], "n_questions_asked": int}
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.task and not self.task.done():
            return "running"
        if self.summary is not None:
            return "done"
        return "none"


audits: dict[str, AuditState] = {}


_vllm_clients: dict[int, AsyncOpenAI] = {}


def vllm_client(model_id: str) -> AsyncOpenAI:
    """Cached AsyncOpenAI for `model_id`'s vLLM port. One httpx pool per port."""
    sup = supervisor()
    port = sup.port_for(model_id)
    if port is None:
        from .supervisor import ModelNotLoaded
        raise ModelNotLoaded(model_id)
    client = _vllm_clients.get(port)
    if client is None:
        client = AsyncOpenAI(
            base_url=f"http://{cfg.server.host}:{port}/v1",
            api_key="EMPTY",
            timeout=cfg.audit.ask_model_timeout_s,
        )
        _vllm_clients[port] = client
    return client


def audit_running_other(model_id: str) -> bool:
    """True if any audit other than `model_id`'s is currently running."""
    return any(sid != model_id and st.status == "running" for sid, st in audits.items())


def model_max_len(model_cfg) -> int:
    """Effective context window for a model (per-model override or global)."""
    return model_cfg.max_model_len if model_cfg.max_model_len is not None else cfg.server.max_model_len


def clamp_max_tokens(model_cfg, prompt, requested: int, safety_margin: int = 32) -> int:
    """Return a safe `max_tokens` value for `prompt`.

    `prompt` may be a string, a messages list, or a pre-encoded list[int] of
    token IDs. For token IDs we know the exact length; for text we use a
    conservative chars/3 heuristic (over-counts so we always clamp *down*).
    Avoids vLLM 400s like 'maximum context length is 2048 ... total of 2049'.
    """
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        est_prompt = len(prompt)
    else:
        text = prompt if isinstance(prompt, str) else "\n".join(
            m.get("content", "") for m in prompt
        )
        est_prompt = max(1, len(text) // 3)
    budget = model_max_len(model_cfg) - est_prompt - safety_margin
    if budget < 16:
        return 16  # let vLLM 400 cleanly rather than silently producing 0 tokens
    return max(1, min(requested, budget))


# Lazily-imported to avoid circular import (supervisor -> state -> supervisor).
_supervisor = None


def supervisor():
    global _supervisor
    if _supervisor is None:
        from .supervisor import Supervisor
        _supervisor = Supervisor(cfg.server)
    return _supervisor
