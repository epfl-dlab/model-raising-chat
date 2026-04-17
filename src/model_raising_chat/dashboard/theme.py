"""Visual primitives — palette, light/dark toggle, semantic helper classes.

Design language is Apple-ish: generous whitespace, hairline dividers, system
font stack, restrained colour. All colours come from CSS custom properties so
toggling light/dark requires only flipping a single attribute on <html>.
"""
from __future__ import annotations

from nicegui import app, ui


# Legacy constants kept for any old call sites — they map to the new helpers.
BG_PAGE = "surface-page"
BG_CARD = "surface"
BG_CARD_HOVER = "surface-hover"
BG_HEADER = "surface-header"
BORDER = "border-divider"
TEXT_DIM = "text-dim"
TEXT_MUTED = "text-muted"
TEXT_NORM = "text-primary"
TEXT_BRIGHT = "text-primary"
ACCENT = "primary"


_THEME_CSS = """
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
               "Inter", "Helvetica Neue", "Segoe UI", system-ui, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  --r-sm: 6px;
  --r-md: 10px;
  --r-lg: 14px;
  --r-pill: 9999px;
  --tx: 140ms cubic-bezier(.4,0,.2,1);
}

html[data-theme="light"] {
  --bg: #f5f5f7;
  --surface: #ffffff;
  --surface-2: #fbfbfd;
  --surface-3: #ececef;
  --surface-hover: rgba(0,0,0,0.045);
  --border: rgba(0,0,0,0.09);
  --border-strong: rgba(0,0,0,0.18);
  --text-primary: #1d1d1f;
  --text-secondary: #3a3a3c;
  --text-dim: #6e6e73;
  --text-muted: #86868b;
  --accent: #0071e3;
  --accent-2: #5856d6;
  --accent-soft: rgba(0,113,227,0.10);
  --success: #1f883d;
  --success-soft: rgba(31,136,61,0.10);
  --warning: #b86e00;
  --warning-soft: rgba(184,110,0,0.10);
  --danger: #d70015;
  --danger-soft: rgba(215,0,21,0.10);
  --info: #0071e3;
  --info-soft: rgba(0,113,227,0.10);
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 6px 22px -10px rgba(0,0,0,0.10);
}

html[data-theme="dark"] {
  --bg: #000000;
  --surface: #1c1c1e;
  --surface-2: #2c2c2e;
  --surface-3: #3a3a3c;
  --surface-hover: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.10);
  --border-strong: rgba(255,255,255,0.22);
  --text-primary: #f5f5f7;
  --text-secondary: #d1d1d6;
  --text-dim: #a1a1a6;
  --text-muted: #6e6e73;
  --accent: #2997ff;
  --accent-2: #5e5ce6;
  --accent-soft: rgba(41,151,255,0.14);
  --success: #30d158;
  --success-soft: rgba(48,209,88,0.14);
  --warning: #ff9f0a;
  --warning-soft: rgba(255,159,10,0.14);
  --danger: #ff453a;
  --danger-soft: rgba(255,69,58,0.14);
  --info: #64d2ff;
  --info-soft: rgba(100,210,255,0.14);
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.30);
  --shadow-md: 0 10px 28px -10px rgba(0,0,0,0.55);
}

html, body {
  background: var(--bg);
  color: var(--text-primary);
  font-family: var(--font-sans);
  letter-spacing: -0.005em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-feature-settings: "rlig" 1, "calt" 1;
}

.nicegui-content { padding: 0; background: var(--bg); }
.q-page { background: var(--bg) !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 5px; border: 2px solid var(--bg); }
::-webkit-scrollbar-track { background: transparent; }

.mono { font-family: var(--font-mono); font-feature-settings: "ss01" 1; font-size: 0.95em; }

/* Surfaces */
.surface { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg); }
.surface-flat { background: var(--surface); }
.surface-2 { background: var(--surface-2); }
.surface-3 { background: var(--surface-3); }
.surface-page { background: var(--bg); }
.surface-header {
  background: color-mix(in srgb, var(--bg) 78%, transparent);
  backdrop-filter: saturate(180%) blur(22px);
  -webkit-backdrop-filter: saturate(180%) blur(22px);
  border-bottom: 1px solid var(--border);
}
.surface-hover { transition: background var(--tx); }
.surface-hover:hover { background: var(--surface-hover); }

.card-elevated {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  box-shadow: var(--shadow-sm);
  transition: transform var(--tx), box-shadow var(--tx), border-color var(--tx);
}
.card-elevated:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}
.card-flat {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
}
.card-accent { border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent); }

/* Dividers */
.divider { background: var(--border); }
.hairline { border-top: 1px solid var(--border); }
.border-divider { border-color: var(--border) !important; }
.border-divider-strong { border-color: var(--border-strong) !important; }
.border-accent { border-color: var(--accent) !important; }
.border-success { border-color: var(--success) !important; }
.border-warning { border-color: var(--warning) !important; }
.border-danger { border-color: var(--danger) !important; }

/* Text */
.text-primary { color: var(--text-primary) !important; }
.text-secondary { color: var(--text-secondary) !important; }
.text-dim { color: var(--text-dim) !important; }
.text-muted { color: var(--text-muted) !important; }
.text-accent { color: var(--accent) !important; }
.text-success { color: var(--success) !important; }
.text-warning { color: var(--warning) !important; }
.text-danger { color: var(--danger) !important; }
.text-info { color: var(--info) !important; }
.eyebrow {
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-muted);
}

/* Chips / badges */
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: var(--r-pill);
  font-size: 11px; font-weight: 600; letter-spacing: 0.01em;
  background: var(--surface-2); color: var(--text-secondary);
  border: 1px solid var(--border);
  white-space: nowrap;
}
.chip-accent  { background: var(--accent-soft);  color: var(--accent);  border-color: transparent; }
.chip-success { background: var(--success-soft); color: var(--success); border-color: transparent; }
.chip-warning { background: var(--warning-soft); color: var(--warning); border-color: transparent; }
.chip-danger  { background: var(--danger-soft);  color: var(--danger);  border-color: transparent; }
.chip-info    { background: var(--info-soft);    color: var(--info);    border-color: transparent; }
.chip-muted   { background: var(--surface-2);    color: var(--text-muted); }

/* Generic input polish */
.q-field--outlined .q-field__control {
  border-radius: var(--r-md) !important;
}
.q-field--outlined .q-field__control:before { border-color: var(--border) !important; }
.q-field--outlined .q-field__control:hover:before { border-color: var(--border-strong) !important; }

/* Backwards-compat shim for any legacy slate-* classes still in pages */
.bg-slate-950, .bg-slate-900, .bg-slate-900\\/80, .bg-slate-900\\/50, .bg-slate-950\\/90 { background: var(--surface) !important; }
.bg-slate-800, .bg-slate-800\\/60, .bg-slate-800\\/70 { background: var(--surface-2) !important; }
.text-slate-100, .text-slate-200 { color: var(--text-primary) !important; }
.text-slate-300 { color: var(--text-secondary) !important; }
.text-slate-400 { color: var(--text-dim) !important; }
.text-slate-500, .text-slate-600, .text-slate-700 { color: var(--text-muted) !important; }
.border-slate-800, .border-slate-700, .border-slate-900 { border-color: var(--border) !important; }

/* Special-token highlight (rendered inside markdown bubbles) */
.special-tok {
  display: inline-block;
  padding: 1px 6px;
  margin: 0 1px;
  border-radius: 4px;
  background: var(--accent-soft);
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 0.88em;
  font-weight: 600;
  letter-spacing: -0.01em;
  border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
  vertical-align: baseline;
  line-height: 1.3;
}
/* Inside an accent-coloured user bubble, flip the chip to a frosted look */
.bubble-user .special-tok {
  background: rgba(255,255,255,0.20);
  color: white;
  border-color: rgba(255,255,255,0.40);
}

/* Truncation utilities */
.truncate-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.truncate-3 { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.badge-pill { padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
"""


def _current_theme() -> str:
    """Return 'light' or 'dark'. Defaults to dark if storage isn't available yet."""
    try:
        v = app.storage.user.get("theme_mode")
        return v if v in ("light", "dark") else "dark"
    except Exception:
        return "dark"


def setup_theme() -> None:
    """Inject palette + apply current theme. Call once per page render."""
    mode = _current_theme()
    # Pair Quasar's dark mode with ours so native components (notify, dialogs) match.
    if mode == "dark":
        ui.dark_mode().enable()
    else:
        ui.dark_mode().disable()
    ui.colors(
        primary="#0071e3" if mode == "light" else "#2997ff",
        secondary="#5856d6",
        accent="#0071e3" if mode == "light" else "#2997ff",
        positive="#1f883d" if mode == "light" else "#30d158",
        negative="#d70015" if mode == "light" else "#ff453a",
        warning="#b86e00" if mode == "light" else "#ff9f0a",
        info="#0071e3" if mode == "light" else "#64d2ff",
    )
    ui.add_head_html(f"<style>{_THEME_CSS}</style>")
    ui.add_body_html(
        f"<script>document.documentElement.setAttribute('data-theme', {mode!r});</script>"
    )


def toggle_theme() -> None:
    """Flip between light/dark and persist the preference."""
    new_mode = "light" if _current_theme() == "dark" else "dark"
    try:
        app.storage.user["theme_mode"] = new_mode
    except Exception:
        pass
    if new_mode == "dark":
        ui.dark_mode().enable()
    else:
        ui.dark_mode().disable()
    ui.run_javascript(
        f"document.documentElement.setAttribute('data-theme', {new_mode!r});"
    )


def theme_toggle_button():
    """A small icon button that flips the theme. Updates its own icon on click."""
    is_dark = _current_theme() == "dark"
    btn = ui.button(icon="light_mode" if is_dark else "dark_mode")
    btn.props("flat dense round").classes("text-secondary").tooltip("Toggle light / dark")

    def _on_click():
        toggle_theme()
        btn.props(f"icon={'light_mode' if _current_theme() == 'dark' else 'dark_mode'}")

    btn.on("click", _on_click)
    return btn


def page_container():
    """Standard page wrapper: centered, max-width, generous padding."""
    return ui.column().classes("w-full max-w-6xl mx-auto px-8 py-8 gap-6")


def model_family(model_id: str) -> str:
    """Heuristic family label for grouping cards on the home grid."""
    parts = model_id.split("_")
    if not parts:
        return "other"
    if parts[0] == "epe" and len(parts) >= 2:
        return f"epe / {parts[1]}"
    if parts[0] == "vanilla" and len(parts) >= 2:
        return f"vanilla / {parts[1]}"
    if parts[0] == "baseline":
        return "baselines"
    return parts[0]
