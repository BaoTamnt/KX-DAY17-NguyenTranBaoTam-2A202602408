from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / ".env", override=False)

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = _provider_config(
        provider=provider,
        model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=_float_env("LLM_TEMPERATURE", 0.2),
        key_env=os.getenv("LLM_API_KEY_ENV"),
        key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    judge_provider = os.getenv("JUDGE_PROVIDER", model.provider)
    judge_model = _provider_config(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", model.model_name),
        temperature=_float_env("JUDGE_TEMPERATURE", 0.0),
        key_env=os.getenv("JUDGE_API_KEY_ENV"),
        key=os.getenv("JUDGE_API_KEY"),
        base_url=os.getenv("JUDGE_BASE_URL"),
    )

    threshold = _int_env("COMPACT_THRESHOLD_TOKENS", 1200)
    keep_messages = _int_env("COMPACT_KEEP_MESSAGES", 6)
    if threshold < 1:
        raise ValueError("COMPACT_THRESHOLD_TOKENS must be greater than zero.")
    if keep_messages < 0:
        raise ValueError("COMPACT_KEEP_MESSAGES cannot be negative.")

    state_dir = root / os.getenv("STATE_DIR", "state")
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep_messages,
        model=model,
        judge_model=judge_model,
    )


def _provider_config(
    provider: str,
    model_name: str,
    temperature: float,
    key_env: str | None = None,
    key: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    from model_provider import normalize_provider

    normalized = normalize_provider(provider)
    env_names = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "",
        "openrouter": "OPENROUTER_API_KEY",
    }
    resolved_key = key or (os.getenv(key_env) if key_env else None)
    if resolved_key is None and env_names[normalized]:
        resolved_key = os.getenv(env_names[normalized])
    base_url_env = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL"}.get(
        normalized
    )
    resolved_base_url = base_url or (os.getenv(base_url_env) if base_url_env else None)
    return ProviderConfig(
        provider=normalized,
        model_name=model_name,
        temperature=temperature,
        api_key=resolved_key,
        base_url=resolved_base_url or None,
    )


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer; got {value!r}.") from error


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number; got {value!r}.") from error
