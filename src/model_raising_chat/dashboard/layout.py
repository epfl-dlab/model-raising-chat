"""Shared header showing GPU state across all loaded models."""
from __future__ import annotations

import re
import time

from nicegui import app, ui

from ..state import audits, registry, supervisor
from .theme import setup_theme, theme_toggle_button


_PCT_RE = re.compile(r"(\d{1,3})\s*%")
_LOAD_HINT_RE = re.compile(r"(?i)(loading|download|shard|safetensors|init|warmup|ready)")


def _running_audit_id() -> str | None:
    for sid, st in audits.items():
        if st.status == "running":
            return sid
    return None


def _last_progress_pct(lines: list[str]) -> int | None:
    for line in reversed(lines):
        m = _PCT_RE.search(line)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except ValueError:
                pass
    return None


def _last_relevant_line(lines: list[str]) -> str:
    for line in reversed(lines):
        if _LOAD_HINT_RE.search(line):
            return line[-160:]
    return (lines[-1][-160:] if lines else "").strip()


# model_id -> wall-clock when it started loading (for the progress bar timer)
_loading_started: dict[str, float] = {}


def _model_name(mid: str) -> str:
    try:
        return registry.get(mid).name
    except KeyError:
        return mid


def header() -> None:
    setup_theme()
    sup = supervisor()

    # ---------- left drawer: full per-model loaded panel ----------
    drawer_open = bool(app.storage.user.get("loaded_drawer_open", False))
    drawer = ui.left_drawer(value=drawer_open, fixed=True, bordered=True).props(
        "breakpoint=0"
    ).classes("p-5 surface-flat gap-3").style(
        "width: 320px; border-right: 1px solid var(--border);"
    )

    def _toggle_drawer():
        drawer.toggle()
        try:
            app.storage.user["loaded_drawer_open"] = drawer.value
        except Exception:
            pass

    @ui.refreshable
    def loaded_panel():
        loaded = sup.loaded_ids()
        starting = sup.starting_ids()
        used = sup.total_gmu()
        cap = sup.server.gpu_memory_utilization_cap
        ratio = (used / cap) if cap else 0.0

        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("memory", size="sm").classes("text-accent")
            ui.label("On GPU").classes("eyebrow")
            ui.space()
            ui.label(f"{len(loaded)}").classes("text-xs text-muted mono")

        ui.linear_progress(value=min(ratio, 1.0), show_value=False).props(
            f"instant-feedback color={'warning' if ratio > 0.85 else 'positive'} size=4px rounded"
        )
        ui.label(f"{used:.0%} of {cap:.0%} cap").classes("text-[11px] text-muted mono")

        if not loaded and not starting:
            with ui.column().classes("w-full items-center justify-center gap-2 py-10 opacity-70"):
                ui.icon("power_settings_new", size="lg").classes("text-muted")
                ui.label("No models loaded").classes("text-sm text-dim")
                ui.button("Browse models", icon="grid_view",
                          on_click=lambda: ui.navigate.to("/models")
                          ).props("flat dense").classes("text-secondary")
            return

        for mid in starting:
            log_tail = sup.recent_log(mid, 60)
            pct = _last_progress_pct(log_tail)
            line = _last_relevant_line(log_tail)
            elapsed = int(time.time() - _loading_started.setdefault(mid, time.time()))
            with ui.card().classes("w-full p-3 card-flat gap-1.5").style(
                "border-color: var(--warning); box-shadow: none;"
            ):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.spinner(size="sm", color="warning")
                    ui.label(_model_name(mid)).classes(
                        "text-[13px] font-semibold text-primary truncate flex-1"
                    )
                    ui.label(f"{elapsed}s").classes("text-[11px] mono text-muted")
                ui.linear_progress(
                    value=(pct / 100) if pct is not None else None, show_value=False
                ).props("instant-feedback color=warning size=3px rounded")
                if line:
                    ui.label(line).classes("text-[10px] mono text-muted truncate")

        for stale in [k for k in _loading_started if k not in starting]:
            _loading_started.pop(stale, None)

        for mid in loaded:
            try:
                cfg = registry.get(mid)
            except KeyError:
                continue
            in_flight = sup.in_flight_for(mid)
            port = sup.port_for(mid)
            audit_running_here = mid in audits and audits[mid].status == "running"
            with ui.card().classes("w-full p-3 card-flat gap-1.5").style(
                "border-color: var(--success); box-shadow: none;"
            ):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.icon("circle", size="2xs").classes("text-success")
                    ui.icon("smart_toy" if cfg.is_chat else "edit_note", size="sm").classes(
                        "text-accent" if cfg.is_chat else "text-secondary"
                    )
                    ui.label(cfg.name).classes(
                        "text-[13px] font-semibold text-primary truncate flex-1"
                    ).tooltip(cfg.name)
                with ui.row().classes("w-full items-center gap-1.5"):
                    ui.label(f":{port}").classes("text-[11px] mono text-muted")
                    ui.label("·").classes("text-[11px] text-muted")
                    if in_flight > 0:
                        ui.label(f"{in_flight} in-flight").classes("text-[11px] text-warning")
                    elif audit_running_here:
                        ui.label("auditing").classes("text-[11px] text-warning")
                    else:
                        ui.label("idle").classes("text-[11px] text-muted")
                with ui.row().classes("w-full gap-1 mt-1 items-center"):
                    ui.button(icon="forum",
                              on_click=lambda mid=mid: ui.navigate.to(f"/chat/{mid}")
                              ).props("flat dense round size=sm").classes("text-accent").tooltip("Chat")
                    ui.button(icon="science",
                              on_click=lambda mid=mid: ui.navigate.to(f"/audit/{mid}")
                              ).props("flat dense round size=sm").classes("text-secondary").tooltip("Audit")
                    ui.space()

                    async def _unload(mid=mid):
                        try:
                            await sup.unload(mid)
                            ui.notify(f"unloaded {mid}", type="info")
                        except Exception as e:
                            ui.notify(f"unload failed: {e}", type="negative")
                        loaded_panel.refresh()

                    ui.button(icon="close", on_click=_unload).props(
                        "flat dense round size=sm"
                    ).classes("text-danger").tooltip(
                        "Unload from GPU"
                    ).set_enabled(not audit_running_here and in_flight == 0)

    with drawer:
        loaded_panel()

    # ---------- top header bar ----------
    with ui.header().classes(
        "surface-header items-center px-6 py-3 gap-4"
    ).style("min-height: 56px; box-shadow: none;"):
        ui.button(icon="menu", on_click=_toggle_drawer).props(
            "flat dense round"
        ).classes("text-secondary").tooltip("Loaded models")

        with ui.row().classes("items-center gap-2 cursor-pointer").on(
            "click", lambda: ui.navigate.to("/")
        ):
            ui.icon("blur_on", size="md").classes("text-accent")
            ui.label("Model Raising").classes(
                "font-semibold text-[15px] text-primary tracking-tight"
            )
            ui.label("playground").classes("eyebrow")

        with ui.row().classes("items-center gap-1 ml-3"):
            ui.button(icon="grid_view", on_click=lambda: ui.navigate.to("/models")).props(
                "flat dense round"
            ).classes("text-secondary").tooltip("All models")
            ui.button(icon="forum", on_click=lambda: ui.navigate.to("/")).props(
                "flat dense round"
            ).classes("text-secondary").tooltip("Loaded model's chat")

        ui.space()

        @ui.refreshable
        def loaded_chip():
            loaded = sup.loaded_ids()
            starting = sup.starting_ids()
            cap = sup.server.gpu_memory_utilization_cap
            used = sup.total_gmu()
            ratio = (used / cap) if cap else 0.0

            cls = (
                "chip chip-warning" if ratio > 0.85
                else "chip chip-success" if loaded
                else "chip chip-muted"
            )
            with ui.row().classes(f"{cls} items-center cursor-pointer").on(
                "click", _toggle_drawer
            ).tooltip("Open loaded-models panel"):
                if starting:
                    ui.spinner(size="xs", color="warning")
                else:
                    ui.icon("memory" if loaded else "power_settings_new", size="xs")
                ui.label(
                    f"{len(loaded)} loaded" if not starting
                    else f"{len(loaded)} loaded · {len(starting)} loading"
                )
                ui.label(f"· {used:.0%} / {cap:.0%}").classes("text-muted")

        loaded_chip()

        @ui.refreshable
        def audit_chip():
            aid = _running_audit_id()
            if aid:
                with ui.row().classes("chip chip-warning items-center ml-1"):
                    ui.spinner(size="xs", color="warning")
                    ui.label("audit")
                    ui.label(_model_name(aid)).classes("text-secondary").style(
                        "max-width: 22ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    )
                    ui.button(
                        icon="open_in_new",
                        on_click=lambda mid=aid: ui.navigate.to(f"/audit/{mid}"),
                    ).props("flat dense round size=sm").classes("text-secondary").tooltip(
                        "View audit"
                    )

        audit_chip()

        theme_toggle_button()

        ui.timer(1.5, lambda: (
            loaded_chip.refresh(),
            audit_chip.refresh(),
            progress_bar.refresh(),
            loaded_panel.refresh(),
        ))

    @ui.refreshable
    def progress_bar():
        starting = sup.starting_ids()
        if not starting:
            _loading_started.clear()
            return
        for stale in list(_loading_started):
            if stale not in starting:
                _loading_started.pop(stale, None)

        with ui.column().classes(
            "w-full surface-2 px-6 py-2 gap-2"
        ).style("border-bottom: 1px solid var(--border);"):
            for sid in starting:
                if sid not in _loading_started:
                    _loading_started[sid] = time.time()
                elapsed = time.time() - _loading_started[sid]
                log_tail = sup.recent_log(sid, 60)
                pct = _last_progress_pct(log_tail)
                line = _last_relevant_line(log_tail)
                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.spinner(size="sm", color="warning")
                        ui.label(f"loading {_model_name(sid)}").classes(
                            "text-sm font-medium text-primary"
                        )
                        ui.label(f"({sid})").classes("text-xs mono text-muted")
                        ui.space()
                        ui.label(f"{int(elapsed)}s").classes("text-xs mono text-dim")
                    if pct is not None:
                        ui.linear_progress(value=pct / 100, show_value=False).props(
                            "instant-feedback color=warning size=4px rounded"
                        )
                        ui.label(f"{pct}%").classes("text-xs mono text-warning")
                    else:
                        ui.linear_progress(value=None, show_value=False).props(
                            "instant-feedback color=warning size=4px rounded"
                        )
                    if line:
                        ui.label(line).classes("text-xs mono text-muted truncate")

    progress_bar()
