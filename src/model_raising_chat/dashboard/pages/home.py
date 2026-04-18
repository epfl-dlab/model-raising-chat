"""Model grid + add-model form."""
from __future__ import annotations

from collections import defaultdict

from nicegui import ui

from ...config import ModelCfg
from ...state import audits, registry, supervisor
from ...supervisor import GpuBudgetExceeded
from ..theme import model_family, page_container


STATUS_STYLE = {
    "none":    {"chip": "chip chip-muted",   "icon": "radio_button_unchecked", "label": "no audit"},
    "running": {"chip": "chip chip-warning", "icon": "autorenew",              "label": "auditing"},
    "done":    {"chip": "chip chip-success", "icon": "check_circle",           "label": "audited"},
    "error":   {"chip": "chip chip-danger",  "icon": "error",                  "label": "audit error"},
}


def _render_card(parent: ui.element, m: ModelCfg) -> None:
    sup = supervisor()
    is_loaded = sup.is_loaded(m.id)
    is_starting = sup.is_starting(m.id)
    st = audits.get(m.id)
    audit_status = st.status if st else "none"
    summary = st.summary if st else None
    audit_running_here = st is not None and st.status == "running"

    type_label = "chat" if m.is_chat else "base"
    type_chip = "chip chip-accent" if m.is_chat else "chip"
    sty = STATUS_STYLE[audit_status]

    extra_cls = " card-accent" if is_loaded else ""

    with parent:
        with ui.card().classes(
            f"w-80 p-0 card-elevated overflow-hidden{extra_cls}"
        ).style("box-shadow: var(--shadow-sm);"):
            with ui.column().classes("w-full p-5 gap-3"):
                with ui.row().classes("w-full items-start gap-3"):
                    ui.icon(
                        "smart_toy" if m.is_chat else "edit_note",
                        size="md",
                    ).classes("text-accent mt-0.5" if m.is_chat else "text-secondary mt-0.5")
                    with ui.column().classes("flex-1 gap-1 min-w-0"):
                        ui.label(m.name).classes(
                            "text-[15px] font-semibold text-primary leading-snug truncate-2"
                        ).tooltip(m.name)
                        ui.label(m.hf_repo).classes(
                            "text-[11px] mono text-muted truncate"
                        ).tooltip(m.hf_repo)

                with ui.row().classes("gap-1.5 items-center flex-wrap"):
                    ui.label(type_label).classes(type_chip)
                    with ui.row().classes(f"{sty['chip']} items-center"):
                        ui.icon(sty["icon"], size="xs")
                        ui.label(sty["label"])
                    if m.deprecated:
                        with ui.row().classes("chip chip-warning items-center").tooltip(
                            "This checkpoint has a known bug. Prefer the replacement in the same family."
                        ):
                            ui.icon("warning", size="xs")
                            ui.label("deprecated — buggy")
                    if is_loaded:
                        with ui.row().classes("chip chip-success items-center"):
                            ui.icon("circle", size="xs")
                            ui.label("on GPU")
                    elif is_starting:
                        with ui.row().classes("chip chip-warning items-center"):
                            ui.spinner(size="xs", color="warning")
                            ui.label("loading")

                if summary:
                    n_findings = len(summary.get("findings", []) or [])
                    n_q = summary.get("n_questions_asked", "?")
                    canaries = summary.get("canaries_detected", []) or []
                    n_can = len(canaries)
                    n_strong = sum(
                        1 for c in canaries if (c.get("confidence") or "weak").lower() == "strong"
                    )
                    summary_text = summary.get("summary", "")
                    if summary_text:
                        ui.label(summary_text).classes(
                            "text-[13px] text-secondary leading-relaxed truncate-3"
                        )
                    with ui.row().classes("gap-2 items-center"):
                        ui.label(f"{n_findings} findings · {n_q} questions").classes(
                            "text-[11px] text-muted"
                        )
                        if n_can:
                            ui.label(
                                f"{n_can} canaries" + (f" ({n_strong} strong)" if n_strong else "")
                            ).classes("chip chip-info")

            with ui.row().classes(
                "w-full px-5 py-3 gap-2 items-center surface-2"
            ).style("border-top: 1px solid var(--border);"):
                async def _load(cfg=m):
                    ui.notify(f"loading {cfg.id}…", type="info")
                    try:
                        await sup.load(cfg)
                        ui.notify(f"loaded {cfg.id}", type="positive")
                    except GpuBudgetExceeded as e:
                        ui.notify(str(e), type="negative", multi_line=True, classes="multi-line-notification")
                    except Exception as e:
                        ui.notify(f"load failed: {e}", type="negative")
                    refresh_grid()

                async def _unload(cfg=m):
                    try:
                        await sup.unload(cfg.id)
                        ui.notify(f"unloaded {cfg.id}", type="info")
                    except Exception as e:
                        ui.notify(f"unload failed: {e}", type="negative")
                    refresh_grid()

                if is_loaded:
                    ui.button("Chat", icon="forum",
                              on_click=lambda cfg=m: ui.navigate.to(f"/chat/{cfg.id}")
                              ).props("unelevated dense rounded color=primary").classes("flex-1")
                    ui.button(icon="power_settings_new", on_click=_unload
                              ).props("flat dense round"
                              ).classes("text-secondary"
                              ).tooltip("Unload from GPU"
                              ).set_enabled(not audit_running_here)
                elif is_starting:
                    ui.button("Loading…", icon="hourglass_empty"
                              ).props("unelevated dense rounded color=warning").classes("flex-1").set_enabled(False)
                else:
                    ui.button("Load", icon="play_arrow", on_click=_load
                              ).props("unelevated dense rounded color=primary").classes("flex-1")

                ui.button(
                    icon="science",
                    on_click=lambda cfg=m: ui.navigate.to(f"/audit/{cfg.id}"),
                ).props("flat dense round").classes("text-secondary").tooltip("Audit")


def _render_add_form(parent: ui.element) -> None:
    with parent:
        with ui.card().classes("w-full p-6 card-flat").style("box-shadow: none;"):
            with ui.row().classes("w-full items-center gap-2 mb-4"):
                ui.icon("add_circle_outline", size="md").classes("text-accent")
                ui.label("Add model").classes("text-base font-semibold text-primary")
                ui.label("· written to conf/models/{id}.yaml").classes("text-xs text-muted mono")

            with ui.row().classes("w-full gap-3 items-end"):
                id_in = ui.input("id", placeholder="my_model_v1").props("dense outlined")
                name_in = ui.input("name", placeholder="Display name").props(
                    "dense outlined"
                ).classes("flex-1")
                hf_in = ui.input("hf_repo", placeholder="org/model-name").props(
                    "dense outlined"
                ).classes("flex-1")
            with ui.row().classes("gap-5 mt-3 items-center"):
                is_chat_in = ui.checkbox("chat model", value=True).props("color=primary")
                trc_in = ui.checkbox("trust_remote_code", value=False).props("color=primary")
                ui.space()

                def _add():
                    try:
                        cfg = ModelCfg(
                            id=id_in.value.strip(),
                            name=name_in.value.strip() or id_in.value.strip(),
                            hf_repo=hf_in.value.strip(),
                            is_chat=bool(is_chat_in.value),
                            trust_remote_code=bool(trc_in.value),
                        )
                        registry.add(cfg)
                        ui.notify(f"added {cfg.id}", type="positive")
                        id_in.value = name_in.value = hf_in.value = ""
                        refresh_grid()
                    except Exception as e:
                        ui.notify(f"add failed: {e}", type="negative")

                ui.button("Add model", icon="add", on_click=_add).props(
                    "unelevated rounded color=primary"
                )


_grid_container: ui.element | None = None


def refresh_grid() -> None:
    if _grid_container is None:
        return
    _grid_container.clear()
    models = registry.all()
    by_family: dict[str, list] = defaultdict(list)
    for m in models:
        by_family[model_family(m.id)].append(m)
    sup = supervisor()
    n_loaded = len(sup.loaded_ids())
    gmu_used = sup.total_gmu()
    gmu_cap = sup.server.gpu_memory_utilization_cap
    with _grid_container:
        n_audited = sum(1 for m in models if (audits.get(m.id) and audits[m.id].summary is not None))
        with ui.row().classes("w-full items-baseline gap-3"):
            ui.label("Models").classes("text-3xl font-semibold text-primary tracking-tight")
            ui.label(
                f"{len(models)} total · {n_audited} audited · {n_loaded} on GPU "
                f"({gmu_used:.0%} of {gmu_cap:.0%})"
            ).classes("text-sm text-dim")
            ui.space()
            def _rescan():
                from ...audit import hydrate_summaries
                registry.scan()
                hydrate_summaries()
                refresh_grid()
            ui.button(icon="refresh", on_click=_rescan).props(
                "flat dense round"
            ).classes("text-secondary").tooltip("Re-scan conf/models/")

        for family in sorted(by_family):
            members = by_family[family]
            with ui.column().classes("w-full gap-3 mt-3"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(family).classes("eyebrow")
                    ui.label(f"({len(members)})").classes("text-xs text-muted")
                    ui.element("div").classes("flex-1 hairline")
                with ui.row().classes("w-full flex-wrap gap-4"):
                    for m in members:
                        _render_card(ui.element("div").classes("contents"), m)


def page() -> None:
    global _grid_container
    with page_container():
        _grid_container = ui.column().classes("w-full gap-4")
        refresh_grid()
        _render_add_form(ui.column().classes("w-full mt-8"))
        ui.timer(2.5, refresh_grid)
