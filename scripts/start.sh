#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env (auto-export every assignment) without echoing its contents.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  echo "[start] loaded .env"
fi

if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
  echo "WARNING: NGROK_AUTHTOKEN unset (not in env, not in .env) — dashboard will start without a public tunnel." >&2
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found. Install Claude Code and run 'claude login' first." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m model_raising_chat.dashboard.app
else
  exec python -m model_raising_chat.dashboard.app
fi
