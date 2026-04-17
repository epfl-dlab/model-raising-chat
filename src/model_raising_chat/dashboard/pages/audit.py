"""Audit launcher + live log + summary view."""
from __future__ import annotations

import asyncio
import itertools
import time

from nicegui import ui

from ...audit import load_summary, run_audit
from ...state import AuditState, audit_running_other, audits, registry, supervisor
from ..theme import page_container


STATUS_STYLE = {
    "none":    {"chip": "chip chip-muted",   "icon": "radio_button_unchecked", "label": "not started"},
    "running": {"chip": "chip chip-warning", "icon": "autorenew",              "label": "running"},
    "done":    {"chip": "chip chip-success", "icon": "check_circle",           "label": "complete"},
    "error":   {"chip": "chip chip-danger",  "icon": "error",                  "label": "error"},
}


def _safe_notify(message: str, **kw) -> None:
    try:
        ui.notify(message, **kw)
    except Exception:
        print(f"[notify] {message}")


def _stat_card(parent, label: str, value: str, icon: str, accent: str = "secondary"):
    with parent:
        with ui.card().classes("flex-1 p-4 card-flat gap-1").style("box-shadow: none;"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon(icon, size="sm").classes(f"text-{accent}")
                ui.label(label).classes("eyebrow")
            ui.label(value).classes("text-2xl font-semibold text-primary mono tracking-tight")


def page(model_id: str) -> None:
    try:
        m = registry.get(model_id)
    except KeyError:
        with page_container():
            ui.label(f"unknown model: {model_id}").classes("text-danger")
        return

    sup = supervisor()
    state = audits.setdefault(model_id, AuditState())
    if state.summary is None:
        on_disk = load_summary(model_id)
        if on_disk is not None:
            state.summary = on_disk

    with page_container():
        with ui.row().classes("w-full items-center gap-3"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/models")).props(
                "flat dense round"
            ).classes("text-secondary")
            ui.icon("science", size="md").classes("text-accent")
            with ui.column().classes("gap-0 min-w-0 flex-shrink"):
                ui.label("Audit").classes("eyebrow")
                ui.label(m.name).classes(
                    "text-2xl font-semibold text-primary tracking-tight leading-tight truncate"
                )
            ui.space()
            ui.label("chat" if m.is_chat else "base").classes(
                "chip chip-accent" if m.is_chat else "chip"
            )

        # ---------- stat cards ----------
        @ui.refreshable
        def stats():
            sty = STATUS_STYLE[state.status]
            elapsed = ""
            if state.started_at:
                end = state.finished_at or time.time()
                secs = int(end - state.started_at)
                elapsed = f"{secs // 60}:{secs % 60:02d}"
            summary = state.summary or {}
            n_findings = len(summary.get("findings", []) or [])
            n_q = summary.get("n_questions_asked", "—")
            canaries = summary.get("canaries_detected", []) or []
            n_can = len(canaries)
            with ui.row().classes("w-full gap-3"):
                status_color = (
                    "warning" if state.status == "running"
                    else "success" if state.status == "done"
                    else "danger" if state.status == "error"
                    else "secondary"
                )
                _stat_card(ui.element("div").classes("contents"), "status", sty["label"], sty["icon"], status_color)
                _stat_card(ui.element("div").classes("contents"), "elapsed", elapsed or "—", "schedule", "accent")
                _stat_card(ui.element("div").classes("contents"), "questions", str(n_q), "help_outline", "secondary")
                _stat_card(ui.element("div").classes("contents"), "findings", str(n_findings), "lightbulb", "warning")
                _stat_card(ui.element("div").classes("contents"), "canaries", str(n_can), "pets", "info")

        stats()

        # ---------- actions ----------
        with ui.row().classes("w-full gap-3 mt-2 items-center"):
            async def start():
                if state.task and not state.task.done():
                    _safe_notify("audit already running for this model", type="warning")
                    return
                if audit_running_other(model_id):
                    _safe_notify("an audit is running on another model — wait for it", type="warning")
                    return
                try:
                    _safe_notify(f"loading {m.id}…", type="info")
                    await sup.load(m)
                except Exception as e:
                    _safe_notify(f"load failed: {e}", type="negative")
                    return
                state.summary = None
                state.error = None
                state.log_lines.clear()
                state.task = asyncio.create_task(run_audit(m))
                _safe_notify("audit started", type="positive")

            def cancel():
                if state.task and not state.task.done():
                    state.task.cancel()
                    _safe_notify("cancellation requested", type="info")

            is_running = state.task is not None and not state.task.done()
            label = "Re-run audit" if state.summary or state.error else "Start audit"
            ui.button(
                "Cancel" if is_running else label,
                icon=("close" if is_running else "play_arrow"),
                on_click=(cancel if is_running else start),
            ).props(f"unelevated rounded color={'negative' if is_running else 'primary'}")
            ui.label(
                "Spawns a Claude Code (Opus) auditor with category subagents that probe the model in parallel."
            ).classes("text-xs text-muted self-center")

        # ---------- summary ----------
        @ui.refreshable
        def summary_panel():
            if state.error:
                with ui.card().classes("w-full p-5 card-flat gap-2").style(
                    "background: var(--danger-soft); border-color: transparent;"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("error", size="sm").classes("text-danger")
                        ui.label("Audit error").classes("font-semibold text-danger")
                    ui.label(state.error).classes("text-sm text-secondary mono")
                return
            if state.summary is None:
                with ui.card().classes(
                    "w-full p-10 card-flat items-center gap-3"
                ).style("box-shadow: none;"):
                    ui.icon("science", size="xl").classes("text-muted opacity-50")
                    ui.label("No audit yet").classes("text-base text-dim")
                    ui.label("Click Start audit above to spawn the Claude-Code auditor.").classes(
                        "text-xs text-muted"
                    )
                return
            with ui.card().classes("w-full p-6 card-flat gap-3").style("box-shadow: none;"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("description", size="sm").classes("text-success")
                    ui.label("Summary").classes("eyebrow")
                ui.markdown(state.summary.get("summary", "")).classes(
                    "text-[14px] text-primary leading-relaxed"
                )
            canaries = state.summary.get("canaries_detected", []) or []
            with ui.card().classes("w-full p-6 card-flat gap-3").style("box-shadow: none;"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("pets", size="sm").classes("text-info")
                    ui.label(f"Canaries detected ({len(canaries)})").classes("eyebrow")
                    ui.label("emergent memorization probes").classes("text-xs text-muted")
                if not canaries:
                    ui.label(
                        "No canaries surfaced across the auditor's probes."
                    ).classes("text-sm text-dim italic mt-1")
                else:
                    with ui.column().classes("w-full gap-2 mt-1"):
                        for c in canaries:
                            cid = c.get("id", "?")
                            quirk = c.get("quirk", "?")
                            value = c.get("value", "?")
                            evidence = c.get("evidence", "")
                            prompt = c.get("prompt_used", "")
                            conf = (c.get("confidence") or "weak").lower()
                            conf_chip = "chip chip-success" if conf == "strong" else "chip chip-warning"
                            with ui.card().classes("w-full p-4 surface-2 card-flat gap-1").style(
                                "box-shadow: none;"
                            ):
                                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                    ui.label(cid).classes("chip chip-muted mono")
                                    ui.label(quirk).classes("text-sm font-semibold text-primary")
                                    ui.label("→").classes("text-muted")
                                    ui.label(value).classes("text-sm text-info mono")
                                    ui.space()
                                    ui.label(conf).classes(f"{conf_chip} uppercase")
                                if evidence:
                                    ui.label(f"evidence: \u201c{evidence}\u201d").classes(
                                        "text-xs text-secondary italic mt-1"
                                    )
                                if prompt:
                                    ui.label(f"prompt: {prompt}").classes(
                                        "text-xs text-muted mono mt-1"
                                    )

            findings = state.summary.get("findings", []) or []
            if findings:
                with ui.card().classes("w-full p-6 card-flat gap-3").style("box-shadow: none;"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("lightbulb", size="sm").classes("text-warning")
                        ui.label(f"Findings ({len(findings)})").classes("eyebrow")
                    with ui.column().classes("w-full gap-1.5 mt-1"):
                        for f in findings:
                            with ui.row().classes("w-full items-start gap-2"):
                                ui.icon("chevron_right", size="sm").classes("text-warning mt-0.5")
                                ui.label(f).classes("text-sm text-primary leading-relaxed flex-1")

        # ---------- live log ----------
        with ui.expansion("Live log", icon="terminal", value=False).classes(
            "w-full card-flat mt-2"
        ).props("dense").style("box-shadow: none;"):
            log = ui.log(max_lines=2000).classes(
                "w-full h-96 surface-2 mono text-[11px] rounded p-3 text-success"
            )
        seen = {"n": 0}

        def drain():
            n = len(state.log_lines)
            if n > seen["n"]:
                for line in itertools.islice(state.log_lines, seen["n"], n):
                    log.push(line)
                seen["n"] = n
                summary_panel.refresh()
            stats.refresh()  # cheap; keeps elapsed counter live during quiet stretches

        ui.timer(0.4, drain)

        summary_panel()
