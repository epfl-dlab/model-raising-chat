"""NiceGUI entry-point. Mounts pages, opens the ngrok tunnel, registers cleanup."""
from __future__ import annotations

import os
import secrets

from nicegui import app, ui

from ..audit import hydrate_summaries
from ..state import cfg, registry, supervisor
from .layout import header
from .pages import audit as audit_page
from .pages import chat as chat_page
from .pages import home as home_page


_public_url: str | None = None


def _start_ngrok() -> None:
    global _public_url
    if not cfg.ngrok.enabled:
        print("[startup] ngrok disabled; dashboard available locally only")
        return
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        print("[startup] WARNING: NGROK_AUTHTOKEN unset; skipping tunnel")
        return
    try:
        from pyngrok import conf, ngrok
        conf.get_default().auth_token = token
        kw: dict = {}
        if cfg.ngrok.basic_auth:
            kw["basic_auth"] = cfg.ngrok.basic_auth
        tunnel = ngrok.connect(addr=cfg.dashboard.port, proto="http", **kw)
        _public_url = tunnel.public_url
        print(f"[startup] PUBLIC URL: {_public_url}")
        if cfg.ngrok.basic_auth:
            print(f"[startup] basic auth required ({len(cfg.ngrok.basic_auth)} credential(s))")
        else:
            print("[startup] WARNING: tunnel exposed without auth — set ngrok.basic_auth in config to gate it")
    except Exception as e:
        print(f"[startup] ngrok tunnel FAILED: {e}")
        print(f"[startup] dashboard remains available at http://{cfg.dashboard.host}:{cfg.dashboard.port}")


def _stop_ngrok() -> None:
    if not cfg.ngrok.enabled:
        return
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception as e:
        print(f"[shutdown] ngrok kill failed: {e}")


@ui.page("/")
def _home():
    sup = supervisor()
    loaded = sup.loaded_ids()
    if len(loaded) == 1:
        # Single loaded model — jump straight into its chat.
        ui.navigate.to(f"/chat/{loaded[0]}")
        return
    header()
    home_page.page()


@ui.page("/models")
def _models():
    header()
    home_page.page()


@ui.page("/chat/{model_id}")
def _chat(model_id: str):
    header()
    chat_page.page(model_id)


@ui.page("/audit/{model_id}")
def _audit(model_id: str):
    header()
    audit_page.page(model_id)


def _on_startup():
    registry.scan()
    hydrate_summaries()
    _start_ngrok()


async def _on_shutdown():
    print("[shutdown] stopping vllm…")
    try:
        await supervisor().shutdown()
    except Exception as e:
        print(f"[shutdown] supervisor shutdown failed: {e}")
    _stop_ngrok()


app.on_startup(_on_startup)
app.on_shutdown(_on_shutdown)


def main() -> None:
    storage_secret = os.environ.get("DASHBOARD_STORAGE_SECRET") or secrets.token_urlsafe(32)
    ui.run(
        host=cfg.dashboard.host,
        port=cfg.dashboard.port,
        title=cfg.dashboard.title,
        reload=False,
        show=False,
        storage_secret=storage_secret,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
