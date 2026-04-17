# model-raising-chat

Internal chat playground for testing unreleased "model-raising" (novel safety-pretrained) checkpoints on a single A100. Includes a one-click Claude-Code (Opus) auditor that probes each model with ~100 questions across 10 categories and writes a short summary of weird/safer/broken behaviors.

## Stack

- **vLLM** — one OpenAI-compatible subprocess per model on a fixed port. Only one model loaded at a time on the GPU; switches are serialized on an in-flight counter.
- **NiceGUI** — dashboard with per-model chat, audit launcher, and add-model form.
- **claude-agent-sdk** — drives an Opus auditor with category subagents and an in-process MCP `ask_model` tool that maintains true multi-turn conversations with the model under test.
- **OmegaConf + per-file YAML** — model registry; new models are added from the dashboard and persisted to `conf/models/*.yaml`.
- **pyngrok** — public URL on startup, gated by ngrok's built-in OAuth.

## Quickstart

```bash
# 1. Install
uv sync   # or: pip install -e .

# 2. Authenticate Claude Code with your Max subscription (one-time)
claude login

# 3. Start the dashboard (also opens an ngrok tunnel)
NGROK_AUTHTOKEN=... bash scripts/start.sh
```

The startup script prints the public URL. Open it, click **Load** on a model, then **Chat** or **Audit**.

## Adding models

Either drop a YAML file into `conf/models/` (see `tulu3sft_smollm.yaml` for the schema) or use the **Add Model** form in the dashboard footer.

## Layout

```
conf/             # config.yaml + per-model YAMLs
data/             # audit_questions.yaml + audits/{id}.json summaries
logs/             # vllm.log + per-audit Claude logs
src/model_raising_chat/
  config.py       # Pydantic schemas, registry I/O
  supervisor.py   # vLLM subprocess lifecycle
  audit.py        # Claude-Code audit runner + MCP ask_model tool
  state.py        # module-level singletons
  dashboard/
    app.py        # NiceGUI entry-point + ngrok
    pages/        # home, chat, audit
scripts/start.sh
```

See `/dlabscratch1/jminder/.claude/plans/i-want-you-to-async-hamming.md` for the full design notes.
