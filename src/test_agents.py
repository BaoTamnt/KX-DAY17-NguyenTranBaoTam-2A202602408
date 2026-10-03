from __future__ import annotations

from pathlib import Path

from dataclasses import replace

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    config = load_config(Path(__file__).resolve().parent.parent)
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=2,
        model=ProviderConfig("openai", "test-model", 0.0),
        judge_model=ProviderConfig("openai", "test-model", 0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("test-user") == "# User profile\n"
    path = store.write_text("test-user", "# User profile\n- Tên: An\n")
    assert path.name == "User.md"
    assert store.file_size("test-user") > 0
    assert store.edit_text("test-user", "Tên: An", "Tên: Bình")
    assert "Tên: Bình" in store.read_text("test-user")
    assert not store.edit_text("test-user", "Tên: An", "Tên: Cường")


def test_profile_extraction_ignores_questions_and_keeps_latest_correction() -> None:
    assert extract_profile_updates("Mình tên là DũngCT.") == {"Tên": "DũngCT"}
    assert extract_profile_updates("Mình tên là DũngCT?") == {}
    assert extract_profile_updates("Mình tên là An. Mình tên gì?") == {"Tên": "An"}
    updates = extract_profile_updates(
        "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng."
    )
    assert updates["Nơi ở hiện tại"] == "Huế"
    profession = extract_profile_updates(
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    )
    assert profession["Nghề nghiệp hiện tại"] == "MLOps engineer"
    noisy_profession = extract_profile_updates(
        "Mình đùa rằng chuyển sang product manager, nhưng nghề nghiệp hiện tại vẫn là MLOps engineer."
    )
    assert noisy_profession["Nghề nghiệp hiện tại"] == "MLOps engineer"
    assert "Nghề nghiệp hiện tại" not in extract_profile_updates(
        "Đoạn compact này đang làm đúng việc của nó."
    )
    assert "ngắn gọn" in extract_profile_updates(
        "Khi giải thích kỹ thuật, hãy trả lời thành bullet ngắn và có ví dụ thực tế."
    )["Phong cách trả lời"]


def test_question_does_not_replace_a_stored_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("alice", "first", "Đồ uống yêu thích là cà phê sữa đá.")
    agent.reply("alice", "second", "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.")
    assert agent.profile_store.facts("alice")["Đồ uống yêu thích"] == "cà phê sữa đá"


def test_compact_trigger(tmp_path: Path) -> None:
    memory = CompactMemoryManager(threshold_tokens=20, keep_messages=2)
    for index in range(8):
        memory.append("thread", "user", f"message {index} " + "detail " * 20)
    context = memory.context("thread")
    assert memory.compaction_count("thread") > 0
    assert len(context["messages"]) <= 2
    assert context["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    advanced.reply("alice", "advanced-first", "Mình tên là Alice.")
    baseline.reply("alice", "baseline-first", "Mình tên là Alice.")
    advanced_answer = advanced.reply("alice", "advanced-new", "Mình tên gì?")["response"]
    baseline_answer = baseline.reply("alice", "baseline-new", "Mình tên gì?")["response"]

    assert "Alice" in advanced_answer
    assert "Alice" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(12):
        message = f"Turn {index}: " + "long context detail " * 35
        baseline.reply("alice", "baseline-long", message)
        advanced.reply("alice", "advanced-long", message)

    assert advanced.compaction_count("advanced-long") > 0
    assert advanced.prompt_token_usage("advanced-long") < baseline.prompt_token_usage(
        "baseline-long"
    )
