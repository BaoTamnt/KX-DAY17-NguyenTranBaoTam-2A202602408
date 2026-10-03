# Memory Systems Lab

The implementation contains both a deterministic offline mode and an optional
live chat-model mode. `config.py` loads provider, path, and compact-memory
settings; `memory_store.py` handles profiles and conversation compaction.

Run commands from the repository root:

```bash
python src/benchmark.py
python -m pytest src/test_agents.py -v
```

Without model credentials, agents use repeatable offline responses. Set
`LLM_PROVIDER`, `LLM_MODEL`, and the provider's API key in the environment (or
in a root `.env` file) to enable live responses. `LLM_PROVIDER` supports
`openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.

The benchmark runs both `data/conversations.json` and
`data/advanced_long_context.json`, reporting recall, prompt/token estimates,
profile growth, and compaction counts. Runtime profiles are written under
`state/profiles/` and are excluded from Git.
