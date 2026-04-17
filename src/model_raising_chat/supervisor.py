"""vLLM subprocess lifecycle.

Multi-model: one vLLM process per loaded model, each on its own port. Loads
are budgeted against `server.gpu_memory_utilization_cap` (sum of per-model
gmu fractions); over-budget loads raise `GpuBudgetExceeded`. Each entry has
its own in-flight counter; `unload(model_id)` waits for it to drain.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import shutil
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import REPO_ROOT, ModelCfg, ServerCfg


LOG_DIR = REPO_ROOT / "logs"
TEMPLATE_CACHE_DIR = REPO_ROOT / "logs" / "chat_template_cache"
SHM_DIR = Path("/dev/shm")


def _hf_download(repo_id: str, filename: str) -> Path | None:
    """Try to download a file from an HF repo. Returns local Path or None."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        return None
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _read_tokenizer_chat_templates(hf_repo: str) -> tuple[str | None, dict[str, str]]:
    """Return (single_string_template_or_None, {name: template} dict).

    HF's tokenizer_config.json `chat_template` field can be either:
      - a single string (most common) → returned in the first slot
      - a list of {"name": ..., "template": ...} entries (Llama-style "named
        templates", also used by Tulu3-derived checkpoints) → returned as dict
    """
    tc = _hf_download(hf_repo, "tokenizer_config.json")
    if tc is None:
        return None, {}
    try:
        data = json.loads(tc.read_text())
    except (json.JSONDecodeError, OSError):
        return None, {}
    tpl = data.get("chat_template")
    if isinstance(tpl, str) and tpl.strip():
        return tpl, {}
    if isinstance(tpl, list):
        named = {}
        for entry in tpl:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("template"), str):
                named[entry["name"]] = entry["template"]
        return None, named
    return None, {}


def _cache_template(hf_repo: str, suffix: str, contents: str) -> Path:
    TEMPLATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_repo = hf_repo.replace("/", "__")
    p = TEMPLATE_CACHE_DIR / f"{safe_repo}__{suffix}.jinja"
    p.write_text(contents)
    return p


def _resolve_chat_template_from_hf(hf_repo: str, push_log) -> Path | None:
    """Auto-resolve a chat template from the HF repo (common locations only).

    Order:
      1. tokenizer_config.json:chat_template — handles both the string form and
         the named-list form (uses 'default' if present, else first entry).
      2. chat_template.jinja at repo root.
    Repos with templates only under `additional_chat_templates/` are NOT
    auto-picked; set `chat_template: <name>` (named template) or
    `chat_template: additional_chat_templates/foo.jinja` (path) in the YAML.
    """
    single, named = _read_tokenizer_chat_templates(hf_repo)
    if single is not None:
        push_log(f"[supervisor] resolved chat_template from HF tokenizer_config.json ({hf_repo})")
        return _cache_template(hf_repo, "tokenizer", single)
    if named:
        pick = "default" if "default" in named else next(iter(named))
        if len(named) > 1:
            push_log(
                f"[supervisor] WARN: HF tokenizer_config.json for {hf_repo} has multiple "
                f"named templates {sorted(named)}; auto-picked '{pick}'. If the checkpoint "
                f"was trained with a different one, set `chat_template: <name>` in the YAML."
            )
        else:
            push_log(
                f"[supervisor] resolved named chat_template '{pick}' from HF tokenizer_config.json ({hf_repo})"
            )
        return _cache_template(hf_repo, f"tokenizer-{pick}", named[pick])
    cj = _hf_download(hf_repo, "chat_template.jinja")
    if cj is not None:
        push_log(f"[supervisor] resolved chat_template.jinja from HF ({hf_repo})")
        return cj
    return None


def _looks_like_path(s: str) -> bool:
    return ("/" in s) or s.endswith(".jinja") or s.startswith(".")


def _resolve_user_chat_template(value: str, hf_repo: str, push_log) -> Path | None:
    """User-supplied `chat_template` in YAML can be:
       - a NAME of a named template inside tokenizer_config.json
         (e.g. `chat_template: epe` or `chat_template: default`)
       - an absolute or repo-root-relative LOCAL path
       - an HF-repo-relative path (e.g. `additional_chat_templates/epe.jinja`)
    """
    if not _looks_like_path(value):
        _, named = _read_tokenizer_chat_templates(hf_repo)
        if value in named:
            push_log(
                f"[supervisor] using named chat_template '{value}' from HF "
                f"tokenizer_config.json ({hf_repo})"
            )
            return _cache_template(hf_repo, f"tokenizer-{value}", named[value])
        push_log(
            f"[supervisor] WARN: named chat_template '{value}' not found in "
            f"tokenizer_config.json ({hf_repo}); available: {sorted(named) or '<none>'}"
        )
        return None
    p = Path(value)
    if p.is_absolute() and p.exists():
        push_log(f"[supervisor] using local chat template: {p}")
        return p
    if not p.is_absolute():
        candidate = REPO_ROOT / p
        if candidate.exists():
            push_log(f"[supervisor] using local chat template: {candidate}")
            return candidate
    fetched = _hf_download(hf_repo, value)
    if fetched is not None:
        push_log(f"[supervisor] resolved HF-relative chat template: {hf_repo}/{value}")
        return fetched
    push_log(f"[supervisor] WARN: chat_template '{value}' not found locally or in HF repo {hf_repo}")
    return None


class ModelNotLoaded(RuntimeError):
    pass


class GpuBudgetExceeded(RuntimeError):
    pass


@dataclass
class ProcEntry:
    cfg: ModelCfg
    port: int
    gpu_memory_utilization: float
    proc: subprocess.Popen | None = None
    in_flight: int = 0
    ready: bool = False
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    log_reader: asyncio.Task | None = None
    log_fh: Any = None


class Supervisor:
    """Multi-process vLLM supervisor.

    Invariants:
      - Each loaded/loading model has exactly one entry in `entries`.
      - Sum of `entries[*].gpu_memory_utilization` ≤ server.cap (enforced at load).
      - `unload(id)` waits for `entries[id].in_flight` to drain before stopping.
      - `use(id)` raises ModelNotLoaded if the entry is missing or not yet ready.
    """

    def __init__(self, server: ServerCfg) -> None:
        self.server = server
        self.entries: dict[str, ProcEntry] = {}
        self.cond = asyncio.Condition()

    # ---------- public queries ----------

    def is_loaded(self, model_id: str) -> bool:
        e = self.entries.get(model_id)
        return e is not None and e.ready

    def is_starting(self, model_id: str) -> bool:
        e = self.entries.get(model_id)
        return e is not None and not e.ready

    def loaded_ids(self) -> list[str]:
        return [mid for mid, e in self.entries.items() if e.ready]

    def starting_ids(self) -> list[str]:
        return [mid for mid, e in self.entries.items() if not e.ready]

    def total_gmu(self) -> float:
        return sum(e.gpu_memory_utilization for e in self.entries.values())

    def port_for(self, model_id: str) -> int | None:
        e = self.entries.get(model_id)
        return e.port if e else None

    def in_flight_for(self, model_id: str) -> int:
        e = self.entries.get(model_id)
        return e.in_flight if e else 0

    def recent_log(self, model_id: str | None = None, n: int = 80) -> list[str]:
        if model_id is None:
            merged: list[str] = []
            for e in self.entries.values():
                merged.extend(e.log_tail)
            return merged[-n:]
        e = self.entries.get(model_id)
        return list(e.log_tail)[-n:] if e else []

    # ---------- public mutators ----------

    async def load(self, cfg: ModelCfg) -> None:
        """Load `cfg` into a new vLLM subprocess on a fresh port.

        - No-op if already loaded. Awaits readiness if currently starting.
        - Raises GpuBudgetExceeded if the gmu sum would breach the cap.
        - Raises (and cleans up) on subprocess failure.
        """
        async with self.cond:
            existing = self.entries.get(cfg.id)
            if existing is not None:
                while not existing.ready and cfg.id in self.entries:
                    await self.cond.wait()
                    existing = self.entries.get(cfg.id)
                return
            gmu = (
                cfg.gpu_memory_utilization
                if cfg.gpu_memory_utilization is not None
                else self.server.gpu_memory_utilization
            )
            new_total = self.total_gmu() + gmu
            if new_total > self.server.gpu_memory_utilization_cap:
                raise GpuBudgetExceeded(
                    f"Cannot load {cfg.id}: gpu_memory_utilization={gmu:.2f} would push "
                    f"total to {new_total:.2f} (cap {self.server.gpu_memory_utilization_cap:.2f}). "
                    f"Currently loaded: {sorted(self.loaded_ids())}. Unload one first."
                )
            entry = ProcEntry(cfg=cfg, port=self._allocate_port(), gpu_memory_utilization=gmu)
            self.entries[cfg.id] = entry
            self.cond.notify_all()

        # vLLM's `--gpu-memory-utilization` is a per-process target measured
        # against total GPU memory. At startup vLLM checks `free >= util*total`;
        # passing the cumulative sum would falsely fail at the 3rd+ load
        # (siblings already hold most of the budget). Pass per-model own_gmu;
        # vLLM caps its own KV-cache to fit within actual free memory.
        try:
            await self._start_entry(entry, entry.gpu_memory_utilization)
        except BaseException:
            with contextlib.suppress(Exception):
                await self._terminate_entry(entry)
            async with self.cond:
                self.entries.pop(cfg.id, None)
                self.cond.notify_all()
            raise

        async with self.cond:
            entry.ready = True
            self.cond.notify_all()

    async def unload(self, model_id: str) -> None:
        """Wait for in-flight to drain, then stop and remove the entry."""
        async with self.cond:
            while True:
                e = self.entries.get(model_id)
                if e is None:
                    return
                if not e.ready:
                    await self.cond.wait()
                    continue
                if e.in_flight == 0:
                    del self.entries[model_id]
                    self.cond.notify_all()
                    break
                await self.cond.wait()
        await self._terminate_entry(e)
        if not self.entries:
            await self._wait_gpu_drain()
            self._cleanup_shm()

    async def shutdown(self) -> None:
        """Stop every running vLLM. Does NOT wait for in-flight."""
        async with self.cond:
            entries = list(self.entries.values())
            self.entries.clear()
            self.cond.notify_all()
        for e in entries:
            with contextlib.suppress(Exception):
                await self._terminate_entry(e)
        if entries:
            with contextlib.suppress(Exception):
                await self._wait_gpu_drain()
            self._cleanup_shm()

    @contextlib.asynccontextmanager
    async def use(self, model_id: str):
        async with self.cond:
            e = self.entries.get(model_id)
            if e is None or not e.ready:
                raise ModelNotLoaded(model_id)
            e.in_flight += 1
        try:
            yield
        finally:
            async with self.cond:
                e.in_flight -= 1
                self.cond.notify_all()

    # ---------- internals ----------

    def _allocate_port(self) -> int:
        used = {e.port for e in self.entries.values()}
        p = self.server.port
        while p in used:
            p += 1
        return p

    async def _start_entry(self, entry: ProcEntry, cumulative_gmu: float) -> None:
        cfg = entry.cfg
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"vllm-{cfg.id}.log"
        cmd = self._build_cmd(entry, cumulative_gmu)
        self._push(
            entry,
            f"[supervisor] starting {cfg.id}: own gmu={entry.gpu_memory_utilization:.3f}, "
            f"cumulative gmu passed to vLLM={cumulative_gmu:.3f}",
        )
        self._push(entry, f"$ {' '.join(cmd)}")
        log_fh = open(log_path, "ab")
        entry.log_fh = log_fh
        env = os.environ.copy()
        if self.server.attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = self.server.attention_backend
            self._push(entry, f"[supervisor] VLLM_ATTENTION_BACKEND={self.server.attention_backend}")
        r, w = os.pipe()
        entry.proc = subprocess.Popen(
            cmd,
            stdout=w,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        os.close(w)
        entry.log_reader = asyncio.create_task(self._tail_pipe(entry, r, log_fh))
        await self._wait_ready(entry)

    async def _terminate_entry(self, entry: ProcEntry) -> None:
        proc = entry.proc
        if entry.log_reader is not None:
            entry.log_reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await entry.log_reader
            entry.log_reader = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.to_thread(proc.wait, self.server.stop_timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(proc.wait, 10)

    def _build_cmd(self, entry: ProcEntry, gmu_for_vllm: float) -> list[str]:
        s = self.server
        cfg = entry.cfg
        max_len = cfg.max_model_len if cfg.max_model_len is not None else s.max_model_len
        cmd = [
            "vllm", "serve", cfg.hf_repo,
            "--served-model-name", cfg.id,
            "--host", s.host,
            "--port", str(entry.port),
            "--gpu-memory-utilization", str(gmu_for_vllm),
            "--max-model-len", str(max_len),
            "--dtype", s.dtype,
        ]
        if s.enforce_eager:
            cmd.append("--enforce-eager")
        if cfg.trust_remote_code:
            cmd.append("--trust-remote-code")
        tpl_path = self._resolve_chat_template(cfg)
        if tpl_path is not None:
            cmd += ["--chat-template", str(tpl_path)]
        cmd += list(cfg.extra_args)
        return cmd

    def _resolve_chat_template(self, cfg: ModelCfg) -> Path | None:
        """Pick the chat template. Only relevant for chat models."""
        if not cfg.is_chat:
            return None
        push = lambda line: self._push_global(line)  # noqa: E731
        if cfg.chat_template:
            return _resolve_user_chat_template(cfg.chat_template, cfg.hf_repo, push)
        return _resolve_chat_template_from_hf(cfg.hf_repo, push)

    async def _tail_pipe(self, entry: ProcEntry, fd: int, log_fh) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=1 << 20)
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await loop.connect_read_pipe(lambda: protocol, os.fdopen(fd, "rb"))
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    log_fh.write(line)
                    log_fh.flush()
                except Exception:
                    pass
                self._push(entry, line.decode("utf-8", errors="replace").rstrip())
        finally:
            transport.close()
            with contextlib.suppress(Exception):
                log_fh.close()

    def _push(self, entry: ProcEntry, line: str) -> None:
        entry.log_tail.append(line)

    def _push_global(self, line: str) -> None:
        # Chat-template resolution log lines: route to the most recently-added
        # entry's tail if available; otherwise drop. Only used during _build_cmd.
        if self.entries:
            last_entry = next(reversed(self.entries.values()))
            last_entry.log_tail.append(line)

    async def _wait_ready(self, entry: ProcEntry) -> None:
        url = f"http://{self.server.host}:{entry.port}/v1/models"
        deadline = time.monotonic() + self.server.ready_timeout_s
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if entry.proc is None or entry.proc.poll() is not None:
                    rc = entry.proc.returncode if entry.proc else "n/a"
                    raise RuntimeError(f"vllm process for {entry.cfg.id} exited (rc={rc})")
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        names = {m.get("id") for m in r.json().get("data", [])}
                        if entry.cfg.id in names:
                            self._push(entry, f"[supervisor] ready: {entry.cfg.id} on :{entry.port}")
                            return
                except (httpx.RequestError, ValueError):
                    pass
                await asyncio.sleep(1.5)
        raise TimeoutError(
            f"vllm server for {entry.cfg.id} not ready within {self.server.ready_timeout_s}s"
        )

    async def _wait_gpu_drain(self) -> None:
        deadline = time.monotonic() + self.server.gpu_drain_timeout_s
        threshold = self.server.gpu_drain_threshold_mib
        while time.monotonic() < deadline:
            used = await asyncio.to_thread(_nvidia_smi_used_mib)
            if used is None:
                return
            if used < threshold:
                return
            await asyncio.sleep(1.0)

    def _cleanup_shm(self) -> None:
        if not SHM_DIR.exists():
            return
        for p in SHM_DIR.glob("vllm*"):
            with contextlib.suppress(Exception):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)


@functools.lru_cache(maxsize=1)
def total_vram_mib() -> int | None:
    """Sum of memory.total across visible GPUs, in MiB. None if nvidia-smi missing."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    total = 0
    for line in out.stdout.strip().splitlines():
        try:
            total += int(line.strip())
        except ValueError:
            return None
    return total


def _nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    total = 0
    for line in out.stdout.strip().splitlines():
        try:
            total += int(line.strip())
        except ValueError:
            return None
    return total
