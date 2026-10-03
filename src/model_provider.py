from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "openai_compatible": "custom",
        "local": "ollama",
    }
    provider = aliases.get(provider, provider)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        raise ValueError(
            f"Unsupported LLM provider {value!r}; choose one of {', '.join(sorted(supported))}."
        )
    return provider


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        options = {**common, "api_key": config.api_key}
        if provider == "custom":
            if not config.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for the custom provider.")
            options["base_url"] = config.base_url
        return ChatOpenAI(**options)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**common, google_api_key=config.api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**common, api_key=config.api_key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        options = dict(common)
        if config.base_url:
            options["base_url"] = config.base_url
        return ChatOllama(**options)
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(**common, api_key=config.api_key)

    raise AssertionError("Provider validation should make this branch unreachable.")
