"""Local tokenization that preserves the model's registered special tokens.

For our model_raising checkpoints, literal occurrences of `<assistant>` (and
the `<charter_*>` family) must tokenize to their dedicated IDs — not be split
into character-level pieces. Going through vLLM's chat/completions endpoint is
not reliable for this on every release, so for our models we render the chat
template + encode locally (with `split_special_tokens=False`) and send
`prompt=<token_ids>` to /v1/completions. External baselines keep the high-level
client paths.
"""
from __future__ import annotations

import functools

from .config import ModelCfg
from .supervisor import (
    _resolve_chat_template_from_hf,
    _resolve_user_chat_template,
)


def is_ours(cfg: ModelCfg) -> bool:
    """True for model_raising checkpoints; False for external baselines."""
    return not cfg.id.startswith("baseline_")


@functools.lru_cache(maxsize=None)
def get_tokenizer(hf_repo: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_repo)


def _noop(_: str) -> None:
    pass


@functools.lru_cache(maxsize=None)
def _resolve_template_text(hf_repo: str, chat_template: str | None) -> str | None:
    """Return the chat-template TEXT for this model, mirroring supervisor logic."""
    if chat_template:
        p = _resolve_user_chat_template(chat_template, hf_repo, _noop)
    else:
        p = _resolve_chat_template_from_hf(hf_repo, _noop)
    if p is None:
        return None
    try:
        return p.read_text()
    except OSError:
        return None


def encode_chat(cfg: ModelCfg, messages: list[dict]) -> list[int]:
    """Apply chat template + encode preserving registered special tokens."""
    tk = get_tokenizer(cfg.hf_repo)
    tpl = _resolve_template_text(cfg.hf_repo, cfg.chat_template)
    kwargs: dict = {}
    if tpl is not None:
        kwargs["chat_template"] = tpl
    rendered = tk.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return tk.encode(rendered, add_special_tokens=False, split_special_tokens=False)


def encode_text(cfg: ModelCfg, text: str) -> list[int]:
    """Encode a raw prefix preserving registered special tokens."""
    tk = get_tokenizer(cfg.hf_repo)
    return tk.encode(text, add_special_tokens=False, split_special_tokens=False)


@functools.lru_cache(maxsize=None)
def special_tokens(hf_repo: str) -> tuple[str, ...]:
    """All registered special-token strings, longest-first (avoids substring
    collisions when used for inline highlighting)."""
    tk = get_tokenizer(hf_repo)
    s = {t for t in tk.all_special_tokens if t and len(t) > 1}
    return tuple(sorted(s, key=len, reverse=True))


def tokenize_inspect(cfg: ModelCfg, text: str) -> list[tuple[int, str, bool]]:
    """Tokenize `text` and return `(token_id, decoded_str, is_special)` triples.

    Mirrors the encode path used for vLLM (`split_special_tokens=False`) so the
    UI inspector shows exactly what the model receives.
    """
    tk = get_tokenizer(cfg.hf_repo)
    ids = tk.encode(text, add_special_tokens=False, split_special_tokens=False)
    specials = set(tk.all_special_tokens)
    out: list[tuple[int, str, bool]] = []
    for tid in ids:
        s = tk.decode([tid])
        out.append((tid, s, s in specials))
    return out


def highlight_specials(text: str, hf_repo: str) -> str:
    """Wrap each registered special-token occurrence in a styled <span>.

    Output is markdown-safe — the spans pass through markdown2 unchanged and
    render via the `.special-tok` CSS class defined in dashboard/theme.py.
    """
    out = text
    for tok in special_tokens(hf_repo):
        if tok in out:
            escaped = tok.replace("<", "&lt;").replace(">", "&gt;")
            out = out.replace(tok, f'<span class="special-tok">{escaped}</span>')
    return out
