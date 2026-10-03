from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        conversations = json.load(file)
    if not isinstance(conversations, list):
        raise ValueError(f"Expected a list of conversations in {path}.")
    for index, conversation in enumerate(conversations):
        if not isinstance(conversation, dict) or not isinstance(conversation.get("turns"), list):
            raise ValueError(f"Conversation at index {index} in {path} has no turns list.")
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0 if answer.strip() else 0.0
    answer_lower = answer.casefold()
    matched = sum(item.casefold() in answer_lower for item in expected)
    ratio = matched / len(expected)
    if ratio == 1:
        return 1.0
    return 0.5 if matched else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    factual_coverage = sum(item.casefold() in answer.casefold() for item in expected) / max(
        len(expected), 1
    )
    concise = 1.0 if len(answer) <= 500 else max(0.0, 1 - (len(answer) - 500) / 1500)
    return round(0.2 + 0.7 * factual_coverage + 0.1 * concise, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    user_ids = {str(conversation.get("user_id", "default")) for conversation in conversations}
    profile_store = UserProfileStore(config.state_dir / "profiles")
    initial_memory = sum(profile_store.file_size(user_id) for user_id in user_ids)

    for conversation in conversations:
        user_id = str(conversation.get("user_id", "default"))
        thread_id = f"{agent_name}-{conversation.get('id', len(thread_ids))}"
        thread_ids.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for question_index, recall in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{question_index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall.get("question", "")))
            answer = str(result["response"])
            expected = [str(item) for item in recall.get("expected_contains", [])]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_memory = sum(profile_store.file_size(user_id) for user_id in user_ids)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(
            agent.prompt_token_usage(thread_id) for thread_id in thread_ids
        ),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=max(0, final_memory - initial_memory),
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = (
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    for title, dataset_path in suites:
        conversations = load_conversations(dataset_path)
        results: list[BenchmarkRow] = []
        for agent_name, agent_type in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
            with TemporaryDirectory(prefix=f"day17-{agent_name.lower()}-") as state_dir:
                isolated_config = replace(config, state_dir=Path(state_dir))
                agent = agent_type(isolated_config)
                results.append(run_agent_benchmark(agent_name, agent, conversations, isolated_config))
        print(f"\n## {title}\n")
        print(format_rows(results))
        _print_interpretation(results)


def _print_interpretation(rows: list[BenchmarkRow]) -> None:
    baseline, advanced = rows
    prompt_change = advanced.prompt_tokens_processed - baseline.prompt_tokens_processed
    direction = "more" if prompt_change > 0 else "fewer"
    print(
        f"\nInterpretation: Advanced processed {abs(prompt_change)} {direction} prompt tokens "
        f"than Baseline in this suite. Persistent profiles improve cross-session recall "
        f"at the cost of {advanced.memory_growth_bytes} bytes of stored profile data; "
        f"compaction ran {advanced.compactions} time(s)."
    )


if __name__ == "__main__":
    main()
