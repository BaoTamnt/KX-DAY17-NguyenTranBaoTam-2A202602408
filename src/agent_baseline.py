from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """An agent whose conversation history is scoped to one thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        del user_id
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        result = self.langchain_agent.invoke(session.messages)
        response = _response_text(result)
        output_tokens = _output_tokens(result, response)
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)

        facts: dict[str, str] = {}
        for item in session.messages:
            facts.update(extract_profile_updates(item["content"]))
        response = (
            "Mình nhớ trong cuộc trò chuyện này: "
            + "; ".join(f"{key}: {value}" for key, value in facts.items())
            if facts
            else "Mình chưa có thông tin đó trong cuộc trò chuyện hiện tại."
        )
        output_tokens = estimate_tokens(response)
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        model = self.config.model
        if model.provider == "ollama":
            return build_chat_model(model)
        if model.provider == "custom":
            if not model.base_url:
                return None
        elif not model.api_key:
            return None
        return build_chat_model(model)


def _response_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "\n".join(part for part in text_parts if part)
    raise TypeError(f"Model returned unsupported response content: {type(content).__name__}.")


def _output_tokens(result: Any, response: str) -> int:
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict) and isinstance(usage.get("output_tokens"), int):
        return usage["output_tokens"]
    return estimate_tokens(response)
