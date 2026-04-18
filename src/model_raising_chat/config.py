"""Pydantic schemas + filesystem registry I/O for models and global config."""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator


REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
MODELS_DIR = CONF_DIR / "models"


class ModelCfg(BaseModel):
    id: str
    name: str
    hf_repo: str
    is_chat: bool
    trust_remote_code: bool = False
    chat_template: str | None = None
    max_model_len: int | None = None  # overrides server default; null = use global
    # Fraction of total GPU memory this model's vLLM process may use. None →
    # server default. Bump for larger checkpoints (e.g. 7B → 0.4).
    gpu_memory_utilization: float | None = None
    extra_args: list[str] = Field(default_factory=list)
    # Flag kept-around-but-known-bad checkpoints so the UI can warn before use.
    deprecated: bool = False

    @field_validator("id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_\-]+", v):
            raise ValueError("id must match [a-zA-Z0-9_-]+ (used as filename and served-model-name)")
        return v


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    # Base port. The supervisor allocates port, port+1, port+2, ... for each
    # concurrently loaded vLLM process.
    port: int = 8001
    # Default per-model gpu_memory_utilization. Sized for ~5 concurrent
    # SmolLM-class checkpoints on an 80 GiB A100; per-model overrides allowed.
    gpu_memory_utilization: float = 0.18
    # Hard cap on the SUM of gpu_memory_utilization across all loaded models.
    # Any load that would push the total past this is rejected with a clear
    # error rather than silently OOMing the GPU.
    gpu_memory_utilization_cap: float = 0.92
    max_model_len: int = 2048  # SmolLM-class default; per-model override supported
    dtype: str = "bfloat16"
    enforce_eager: bool = True
    # FA2's V1 backend crashes with a CUDA device-side assert on some SmolLM
    # checkpoints. TRITON_ATTN is the safe fallback. Set to null to let vLLM
    # auto-pick (FLASH_ATTN on Ampere). Other valid values: FLASH_ATTN, FLASHINFER.
    attention_backend: str | None = "TRITON_ATTN"
    ready_timeout_s: int = 240
    stop_timeout_s: int = 30
    gpu_drain_timeout_s: int = 60
    gpu_drain_threshold_mib: int = 2000


class DashboardCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    title: str = "Model Raising Playground"


class AuditCfg(BaseModel):
    questions_file: str = "data/audit_questions.yaml"
    canaries_file: str = "data/canaries.yaml"
    out_dir: str = "data/audits"
    scratch_dir: str = "logs/claude_audits"
    model: str = "claude-opus-4-7"
    max_turns: int = 300
    ask_model_max_tokens: int = 1024
    ask_model_timeout_s: int = 120
    # Sampling for ask-model MCP calls. OpenAI client defaults (temperature=1.0)
    # are too hot for SmolLM-class checkpoints; use chat-UI-like defaults.
    ask_model_temperature: float = 0.7
    ask_model_top_p: float = 0.95


class NgrokCfg(BaseModel):
    enabled: bool = True
    # ngrok v3 dropped legacy oauth_provider/oauth_allow_emails fields on
    # HTTPv2Tunnel — OAuth is now configured via Traffic Policy. For now, use
    # basic_auth (still supported) for a one-line gate. Leave empty to expose
    # the tunnel without auth (URLs are random; OK for short internal sessions).
    basic_auth: list[str] = Field(default_factory=list)  # ["user:pass", ...]


class AppCfg(BaseModel):
    server: ServerCfg = Field(default_factory=ServerCfg)
    dashboard: DashboardCfg = Field(default_factory=DashboardCfg)
    audit: AuditCfg = Field(default_factory=AuditCfg)
    ngrok: NgrokCfg = Field(default_factory=NgrokCfg)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppCfg":
        path = path or (CONF_DIR / "config.yaml")
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**raw)  # type: ignore[arg-type]


class ModelRegistry:
    """Per-file YAML registry under conf/models/."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._models: dict[str, ModelCfg] = {}
        self.scan()

    def scan(self) -> None:
        self._models = {}
        if not self.models_dir.exists():
            return
        for path in sorted(self.models_dir.glob("*.yaml")):
            raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            cfg = ModelCfg(**raw)  # type: ignore[arg-type]
            self._models[cfg.id] = cfg

    def all(self) -> list[ModelCfg]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelCfg:
        if model_id not in self._models:
            raise KeyError(f"unknown model: {model_id}")
        return self._models[model_id]

    def add(self, cfg: ModelCfg) -> Path:
        path = self.models_dir / f"{cfg.id}.yaml"
        if path.exists() or cfg.id in self._models:
            raise ValueError(f"model already exists: {cfg.id}")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
        self._models[cfg.id] = cfg
        return path
