from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Short-term, persistent-profile, and compact-memory agent."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        profile = self.profile_store.read_text(user_id)
        prompt = "Persistent user profile:\n" + profile
        if context["summary"]:
            prompt += "\nConversation summary:\n" + str(context["summary"])
        messages = [{"role": "system", "content": prompt}] + list(context["messages"])
        result = self.langchain_agent.invoke(messages)
        response = _response_text(result)
        output_tokens = _output_tokens(result, response)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "response": response,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        output_tokens = estimate_tokens(response)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "response": response,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        content = self.profile_store.read_text(user_id) + str(context["summary"])
        content += "".join(message["content"] for message in context["messages"])
        return estimate_tokens(content)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        del thread_id
        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Mình chưa có thông tin hồ sơ nào được lưu."

        query = message.casefold()
        requested: list[str] = []
        if any(term in query for term in ("tên", "là ai", "who")):
            requested.append("Tên")
        if any(term in query for term in ("ở đâu", "nơi ở", "đang ở", "còn ở", "địa điểm")) or (
            "ở " in query and any(city in query for city in ("huế", "đà nẵng", "hà nội"))
        ):
            requested.append("Nơi ở hiện tại")
        if any(term in query for term in ("nghề", "công việc", "job", "product manager")):
            requested.append("Nghề nghiệp hiện tại")
        if any(term in query for term in ("style", "phong cách", "trả lời", "bullet")):
            requested.append("Phong cách trả lời")
        if any(term in query for term in ("đồ uống", "uống", "cà phê")):
            requested.append("Đồ uống yêu thích")
        if any(term in query for term in ("món ăn", "món ruột", "món yêu thích")):
            requested.append("Món ăn yêu thích")
        if any(term in query for term in ("nuôi", "con gì", "corgi")):
            requested.append("Thú cưng")
        if any(term in query for term in ("quan tâm", "python", "ai ", "mối quan tâm")):
            requested.append("Mối quan tâm")

        answer_facts = [(key, facts[key]) for key in requested if key in facts]
        if not answer_facts:
            return "Mình đã ghi nhớ."
        return "Mình nhớ: " + "; ".join(f"{key}: {value}" for key, value in answer_facts)

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
