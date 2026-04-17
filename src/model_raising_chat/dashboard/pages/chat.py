"""Chat for chat models; prefix-completion for base models."""
from __future__ import annotations

from nicegui import app, ui

from ...state import clamp_max_tokens, registry, supervisor, vllm_client
from ...supervisor import GpuBudgetExceeded, ModelNotLoaded
from ...tokenize import (
    encode_chat,
    encode_text,
    highlight_specials,
    is_ours,
    special_tokens,
    tokenize_inspect,
)
from ..theme import page_container


def _history(model_id: str) -> list[dict]:
    return app.storage.user.setdefault("history", {}).setdefault(model_id, [])


def _token_inspector(m, get_text) -> None:
    """Collapsible card that breaks the textbox into per-token chips on demand.

    Shows token id + decoded string for every token; specials get the accent
    pill style. Lets you verify that `<assistant>` (id 49152 on EPE models)
    really lands on its dedicated id rather than splitting into pieces.
    """
    with ui.expansion("Token inspector", icon="bug_report", value=False).classes(
        "w-full card-flat mt-2"
    ).props("dense").style("box-shadow: none;"):
        with ui.column().classes("w-full p-3 gap-2") as body:
            chips = ui.row().classes("w-full flex-wrap gap-1")
            count = ui.label("").classes("text-xs text-muted mono")

        def _refresh():
            chips.clear()
            text = get_text()
            if not text.strip():
                count.text = "(empty)"
                return
            try:
                triples = tokenize_inspect(m, text)
            except Exception as e:
                count.text = f"error: {type(e).__name__}: {e}"
                return
            count.text = f"{len(triples)} tokens"
            with chips:
                for tid, s, is_special in triples:
                    visible = s.replace("\n", "↵").replace("\t", "→")
                    if not visible:
                        visible = "·"
                    label = f"{visible}  {tid}"
                    cls = "chip chip-accent mono" if is_special else "chip mono"
                    ui.label(label).classes(cls).style("font-size: 11px;")

        ui.button("Refresh", icon="refresh", on_click=_refresh).props(
            "flat dense"
        ).classes("text-secondary self-start")
        _refresh()


def _specials_hint(m) -> None:
    """Small one-line hint listing the special tokens this model recognises."""
    if not is_ours(m):
        return
    try:
        toks = special_tokens(m.hf_repo)
    except Exception:
        return
    if not toks:
        return
    # Filter out the structural ones; keep model-distinctive tokens up front.
    structural = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
    interesting = [t for t in toks if t not in structural]
    shown = interesting[:6]
    extra = len(interesting) - len(shown)
    with ui.row().classes("items-center gap-1.5 flex-wrap text-[11px] text-muted"):
        ui.label("recognised specials:").classes("text-muted")
        for t in shown:
            ui.label(t).classes("special-tok").style("font-size: 10.5px;")
        if extra > 0:
            ui.label(f"+{extra} more").classes("text-muted italic")


def _prefix(model_id: str) -> dict:
    return app.storage.user.setdefault("prefix", {}).setdefault(model_id, {"text": ""})


def page(model_id: str) -> None:
    try:
        m = registry.get(model_id)
    except KeyError:
        with page_container():
            ui.label(f"unknown model: {model_id}").classes("text-danger")
        return

    sup = supervisor()
    sampling = app.storage.user.setdefault(
        "sampling", {"temperature": 0.7, "top_p": 0.95, "max_tokens": 256}
    )

    drawer = ui.right_drawer(value=False, fixed=True).classes("p-6 surface-flat").style(
        "border-left: 1px solid var(--border);"
    )
    with drawer:
        ui.label("Sampling").classes("eyebrow mb-3")
        ui.number(label="temperature", value=sampling["temperature"], step=0.05, min=0, max=2).bind_value(
            sampling, "temperature"
        ).props("dense outlined").classes("mb-3")
        ui.number(label="top_p", value=sampling["top_p"], step=0.05, min=0, max=1).bind_value(
            sampling, "top_p"
        ).props("dense outlined").classes("mb-3")
        ui.number(label="max_tokens", value=sampling["max_tokens"], step=64, min=1, max=8192).bind_value(
            sampling, "max_tokens"
        ).props("dense outlined")

    with page_container():
        with ui.row().classes("w-full items-center gap-3"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/models")).props(
                "flat dense round"
            ).classes("text-secondary")
            ui.icon("smart_toy" if m.is_chat else "edit_note", size="md").classes(
                "text-accent" if m.is_chat else "text-secondary"
            )
            with ui.column().classes("gap-0 min-w-0 flex-shrink"):
                ui.label(m.name).classes(
                    "text-2xl font-semibold text-primary tracking-tight leading-tight truncate"
                )
                ui.label(m.hf_repo).classes("text-xs mono text-muted truncate")
            ui.space()
            ui.label("chat" if m.is_chat else "base").classes(
                "chip chip-accent" if m.is_chat else "chip"
            )

            async def _ensure_loaded():
                ui.notify(f"loading {m.id}…", type="info")
                try:
                    await sup.load(m)
                    ui.notify(f"loaded {m.id}", type="positive")
                except GpuBudgetExceeded as e:
                    ui.notify(str(e), type="negative", multi_line=True, classes="multi-line-notification")
                except Exception as e:
                    ui.notify(f"load failed: {e}", type="negative")
                status_label.refresh()

            async def _unload():
                try:
                    await sup.unload(model_id)
                    ui.notify(f"unloaded {m.id}", type="info")
                except Exception as e:
                    ui.notify(f"unload failed: {e}", type="negative")
                status_label.refresh()

            @ui.refreshable
            def status_label():
                if sup.is_loaded(model_id):
                    with ui.row().classes("chip chip-success items-center"):
                        ui.icon("circle", size="xs")
                        ui.label("loaded")
                    ui.button("Unload", icon="power_settings_new", on_click=_unload).props(
                        "flat dense"
                    ).classes("text-secondary")
                elif sup.is_starting(model_id):
                    with ui.row().classes("chip chip-warning items-center"):
                        ui.spinner(size="xs", color="warning")
                        ui.label("loading")
                else:
                    ui.button("Load", icon="play_arrow", on_click=_ensure_loaded).props(
                        "unelevated dense rounded color=primary"
                    )

            status_label()
            ui.button(icon="tune", on_click=drawer.toggle).props(
                "flat dense round"
            ).classes("text-secondary").tooltip("Sampling")

        ui.element("div").classes("w-full hairline")

        if m.is_chat:
            _chat_ui(m, sampling, status_label)
        else:
            _base_ui(m, sampling, status_label)


# ---------- chat-model UI ----------

def _chat_ui(m, sampling, status_label) -> None:
    sup = supervisor()
    history = _history(m.id)

    msgs_box = ui.column().classes("w-full gap-4 min-h-[40vh]")

    def render_history():
        msgs_box.clear()
        with msgs_box:
            if not history:
                with ui.column().classes("w-full items-center justify-center py-20 gap-3 opacity-70"):
                    ui.icon("forum", size="xl").classes("text-muted")
                    ui.label("Send a message to start chatting").classes("text-sm text-dim")
            for h in history:
                _render_msg(h)

    def _render_msg(h):
        is_user = h["role"] == "user"
        content = highlight_specials(h["content"], m.hf_repo) if is_ours(m) else h["content"]
        with ui.row().classes(
            "w-full " + ("justify-end" if is_user else "justify-start")
        ):
            with ui.column().classes("max-w-[78%] gap-1"):
                ui.label("you" if is_user else "assistant").classes(
                    f"text-[10px] font-semibold uppercase tracking-wider "
                    + ("text-muted text-right" if is_user else "text-accent")
                )
                bubble = ui.card().classes(
                    "px-4 py-3 rounded-2xl " + ("bubble-user" if is_user else "")
                ).style(
                    "box-shadow: none; "
                    + (
                        "background: var(--accent); color: white; border: none;"
                        if is_user
                        else "background: var(--surface-2); border: 1px solid var(--border);"
                    )
                )
                with bubble:
                    md = ui.markdown(content).classes(
                        "text-[14px] leading-relaxed " + ("text-white" if is_user else "text-primary")
                    )
                    if is_user:
                        md.style("color: white;")

    render_history()

    async def send():
        text = input_box.value.strip()
        if not text:
            return
        input_box.value = ""
        history.append({"role": "user", "content": text})
        with msgs_box:
            _render_msg({"role": "user", "content": text})
            with ui.row().classes("w-full justify-start"):
                with ui.column().classes("max-w-[78%] gap-1"):
                    ui.label("assistant").classes(
                        "text-[10px] font-semibold uppercase tracking-wider text-accent"
                    )
                    bubble = ui.card().classes("px-4 py-3 rounded-2xl").style(
                        "box-shadow: none; background: var(--surface-2); "
                        "border: 1px solid var(--border);"
                    )
                    with bubble:
                        md = ui.markdown("…").classes("text-[14px] text-primary leading-relaxed")

        try:
            cm = sup.use(m.id)
            try:
                await cm.__aenter__()
            except ModelNotLoaded:
                md.content = (
                    f"**[unloaded]** `{m.id}` is no longer on the GPU. Click **Load** to bring it back."
                )
                history.pop()
                status_label.refresh()
                return
            try:
                client = vllm_client(m.id)
                full = ""
                msgs = [{"role": h["role"], "content": h["content"]} for h in history]
                if is_ours(m):
                    # Pre-tokenize so `<assistant>` etc. land on their special IDs.
                    # `skip_special_tokens=False` lets the UI surface specials the
                    # model itself emits.
                    token_ids = encode_chat(m, msgs)
                    stream = await client.completions.create(
                        model=m.id,
                        prompt=token_ids,
                        stream=True,
                        temperature=sampling["temperature"],
                        top_p=sampling["top_p"],
                        max_tokens=clamp_max_tokens(m, token_ids, int(sampling["max_tokens"])),
                        extra_body={"skip_special_tokens": False},
                    )
                    async for chunk in stream:
                        delta = (chunk.choices[0].text or "") if chunk.choices else ""
                        if delta:
                            full += delta
                            md.content = highlight_specials(full, m.hf_repo)
                else:
                    stream = await client.chat.completions.create(
                        model=m.id,
                        messages=msgs,
                        stream=True,
                        temperature=sampling["temperature"],
                        top_p=sampling["top_p"],
                        max_tokens=clamp_max_tokens(m, msgs, int(sampling["max_tokens"])),
                    )
                    async for chunk in stream:
                        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                        if delta:
                            full += delta
                            md.content = full
                history.append({"role": "assistant", "content": full})
            finally:
                await cm.__aexit__(None, None, None)
        except Exception as e:
            md.content = f"**[error]** `{type(e).__name__}: {e}`"
            history.append({"role": "assistant", "content": f"[error] {e}"})

    with ui.element("div").classes(
        "w-full sticky bottom-0 surface-header -mx-8 px-8 py-4 mt-6"
    ):
        with ui.column().classes("w-full max-w-6xl mx-auto gap-1"):
            with ui.row().classes("w-full items-end gap-2"):
                input_box = ui.textarea(placeholder="Type a message — Ctrl+Enter to send").props(
                    "autogrow outlined input-class=text-primary"
                ).classes("flex-1")
                input_box.on("keydown.enter", lambda e: e.args.get("ctrlKey") and send())
                ui.button("Send", icon="send", on_click=send).props(
                    "unelevated rounded color=primary"
                )
                ui.button(icon="delete_sweep", on_click=lambda: (history.clear(), render_history())).props(
                    "flat dense round"
                ).classes("text-secondary").tooltip("Clear history")
            _specials_hint(m)


# ---------- base-model UI: pure prefix completion ----------

def _base_ui(m, sampling, status_label) -> None:
    sup = supervisor()
    state = _prefix(m.id)

    with ui.card().classes("w-full p-4 surface-2 card-flat gap-2").style("box-shadow: none;"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("info", size="sm").classes("text-secondary")
            ui.label(
                "Base model — type a prefix and Continue. The model appends a completion. "
                "No turns, no chat template; edit the box freely."
            ).classes("text-xs text-dim italic")

    box = ui.textarea(value=state["text"], placeholder="prefix…").props(
        "autogrow outlined input-class='mono text-sm leading-relaxed text-primary'"
    ).classes("w-full mt-2")
    box.bind_value(state, "text")

    if is_ours(m):
        _token_inspector(m, lambda: box.value or "")

    async def _continue():
        prompt = box.value or ""
        if not prompt:
            ui.notify("nothing to continue — type a prefix first", type="warning")
            return
        try:
            cm = sup.use(m.id)
            try:
                await cm.__aenter__()
            except ModelNotLoaded:
                ui.notify(f"{m.id} is not loaded — click Load and retry", type="warning")
                status_label.refresh()
                return
            try:
                client = vllm_client(m.id)
                # For our models, encode locally so registered specials
                # (e.g. `<assistant>` on EPE base models) land on their IDs.
                # `skip_special_tokens=False` lets the UI show specials the
                # model itself emits in its continuation.
                effective_prompt = encode_text(m, prompt) if is_ours(m) else prompt
                extra: dict = {"skip_special_tokens": False} if is_ours(m) else {}
                stream = await client.completions.create(
                    model=m.id,
                    prompt=effective_prompt,
                    stream=True,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    max_tokens=clamp_max_tokens(m, effective_prompt, int(sampling["max_tokens"])),
                    extra_body=extra,
                )
                async for chunk in stream:
                    delta = (chunk.choices[0].text or "") if chunk.choices else ""
                    if delta:
                        box.value += delta
                        state["text"] = box.value
            finally:
                await cm.__aexit__(None, None, None)
        except Exception as e:
            ui.notify(f"{type(e).__name__}: {e}", type="negative")

    def _clear():
        box.value = ""
        state["text"] = ""

    with ui.element("div").classes(
        "w-full sticky bottom-0 surface-header -mx-8 px-8 py-4 mt-6"
    ):
        with ui.column().classes("w-full max-w-6xl mx-auto gap-1"):
            with ui.row().classes("w-full gap-2 items-center"):
                ui.button("Continue", icon="play_arrow", on_click=_continue).props(
                    "unelevated rounded color=primary"
                )
                ui.button("Clear", icon="delete_sweep", on_click=_clear).props(
                    "flat"
                ).classes("text-secondary")
                ui.label("Continue appends to the end of the box").classes("text-xs text-muted ml-2")
            _specials_hint(m)
