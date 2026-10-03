from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import unicodedata
from pathlib import Path


def estimate_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    words = len(normalized.split())
    return max(words, math.ceil(len(normalized) / 4))


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        normalized = unicodedata.normalize("NFKC", user_id).strip()
        slug = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip(".-")
        if not slug:
            raise ValueError("user_id must contain at least one letter or number.")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([^:]+):\s*(.*)$", line)
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        value = value.strip()
        if not value:
            return
        facts = self.facts(user_id)
        if key == "Mối quan tâm" and key in facts:
            existing = [part.strip() for part in facts[key].split(",") if part.strip()]
            additions = [part.strip() for part in re.split(r",| và ", value) if part.strip()]
            for item in additions:
                if item.casefold() not in {part.casefold() for part in existing}:
                    existing.append(item)
            value = ", ".join(existing)
        facts[key] = value
        lines = ["# User profile", ""]
        lines.extend(f"- {fact_key}: {fact_value}" for fact_key, fact_value in facts.items())
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    normalized = " ".join(message.split())
    segments = re.split(r"(?<=[.!?])\s+", normalized)
    declarative_segments = [
        segment for segment in segments if not _is_question_or_recall_request(segment)
    ]
    if not declarative_segments:
        return {}
    text = " ".join(declarative_segments)

    updates: dict[str, str] = {}
    name = re.search(r"\b(?:mình|tôi)\s+tên\s+(?:là\s+)?([^,.!?;]+)", text, re.I)
    if name:
        updates["Tên"] = name.group(1).strip()

    location_patterns = (
        r"\b(?:hiện tại|hiện)\s+(?:đang\s+)?(?:ở|sống tại)\s+([^,.!?;]+)",
        r"\b(?:hiện tại\s+)?(?:mình|tôi)\s+(?:đang\s+)?(?:làm việc\s+)?(?:ở|sống tại)\s+([^,.!?;]+)",
        r"\b(?:mình|tôi)\s+(?:đang\s+)?làm việc ở\s+([^,.!?;]+)",
    )
    for pattern in location_patterns:
        location = re.search(pattern, text, re.I)
        if location:
            value = _trim_fact_value(location.group(1), (" chứ", " nhưng", " và", " để", " vì", " vài ", " mỗi "))
            if value and not re.search(r"\b(quán|văn phòng|nhà hàng|khách sạn)\b", value, re.I):
                updates["Nơi ở hiện tại"] = value
            break

    profession_patterns = (
        r"\b(?:giờ\s+)?(?:mình\s+)?chuyển sang\s+([^,.!?;]+)",
        r"\bnghề nghiệp(?:\s+hiện tại)?\s+(?:thì\s+)?(?:vẫn\s+)?là\s+([^,.!?;]+)",
        r"\b(?:hiện tại\s+)?(?:mình|tôi)\s+(?:đang\s+)?làm\s+([^,.!?;]+)",
        r"\b(?:và\s+)?đang làm\s+([^,.!?;]+)",
    )
    professions: list[tuple[int, str]] = []
    for pattern in profession_patterns:
        for profession in re.finditer(pattern, text, re.I):
            value = _trim_fact_value(
                profession.group(1), (" cho ", " tại ", " và ", " chứ", " nhưng", " không ", " nữa")
            )
            if value and not re.match(r"(?:đúng\s+)?việc\b", value, re.I):
                professions.append((profession.start(), value))
    if professions:
        _, current_profession = max(professions, key=lambda candidate: candidate[0])
        updates["Nghề nghiệp hiện tại"] = current_profession

    drink = re.search(
        r"\b(?:đồ uống yêu thích(?: của mình)? là|mình thích uống|mình vẫn uống)\s+([^,.!?;]+)",
        text,
        re.I,
    )
    if drink:
        updates["Đồ uống yêu thích"] = _trim_fact_value(
            drink.group(1), (" như cũ", " nhưng", " và ", " để ", " vì ")
        )
    food = re.search(r"\b(?:món ăn yêu thích(?: của mình)? là|món ruột là)\s+([^,.!?;]+)", text, re.I)
    if food:
        updates["Món ăn yêu thích"] = food.group(1).strip()

    pet = re.search(r"\b(?:mình\s+)?nuôi\s+(?:một\s+)?(?:bé\s+)?([^,.!?;]+)", text, re.I)
    if pet:
        updates["Thú cưng"] = pet.group(1).strip()

    style_patterns = (
        r"(?:hãy\s+)?trả lời(?:\s+thành)?\s+([^.!?;]+)",
        r"(?:hãy\s+)?giải thích(?:\s+thành)?\s+([^.!?;]+)",
        r"cách giải thích\s+(?:có\s+)?([^.!?;]+)",
        r"câu trả lời\s+(?:nên\s+|cần\s+|hãy\s+)?([^.!?;]+)",
    )
    styles: list[str] = []
    for pattern in style_patterns:
        for style in re.finditer(pattern, text, re.I):
            value = _trim_fact_value(
                style.group(1),
                (" khi ", " hơn là", " nếu ", " như ", " chứ", " vì ", " và nếu"),
            )
            if re.search(r"\b(ngắn gọn|bullet|rõ ý|ví dụ|3 bullet)\b", value, re.I):
                if re.search(r"\b(?:ngắn|gọn)\b", value, re.I) and not re.search(
                    r"\bngắn gọn\b", value, re.I
                ):
                    value = "ngắn gọn, " + value
                styles.append(value)
    if styles:
        updates["Phong cách trả lời"] = styles[-1]

    interests = re.search(
        r"\b(?:mình|tôi)\s+(?:thích|đang quan tâm(?: nhiều)? đến)\s+([^.!?;]+)",
        text,
        re.I,
    )
    if interests:
        value = _trim_fact_value(
            interests.group(1),
            (" hơn là", " vì ", " nhé", " và cách giải thích", ", cách giải thích"),
        )
        if value and not re.match(r"(?:cách giải thích|style trả lời|câu trả lời)\b", value, re.I):
            updates["Mối quan tâm"] = value

    return {key: value for key, value in updates.items() if value}


def _is_question_or_recall_request(text: str) -> bool:
    text = text.strip()
    if not text or text.endswith("?"):
        return True
    if re.search(r"\b(?:là gì|ở đâu|làm gì|như thế nào|con gì|tên gì)\.?$", text, re.I):
        return True
    return bool(
        re.match(
            r"^(?:(?:bạn|hãy)\s+)?(?:thử\s+)?(?:có thể\s+)?(?:nhắc lại|nhớ lại|có biết)\b",
            text,
            re.I,
        )
        or re.match(r"^sang thread mới\b.*\bnhắc lại\b", text, re.I)
    )


def _trim_fact_value(value: str, stop_phrases: tuple[str, ...]) -> str:
    result = value.strip()
    for phrase in stop_phrases:
        index = result.lower().find(phrase.lower())
        if index >= 0:
            result = result[:index].strip()
    return result.strip(" ,.-")


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if max_items <= 0:
        return ""
    snippets: list[str] = []
    for message in messages:
        text = " ".join(message.get("content", "").split())
        if not text:
            continue
        snippet = f"{message.get('role', 'user')}: {text[:180]}"
        if snippet not in snippets:
            snippets.append(snippet)
    return "\n".join(snippets[-max_items:])[:1800]


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if self.threshold_tokens < 1:
            raise ValueError("threshold_tokens must be greater than zero.")
        if self.keep_messages < 0:
            raise ValueError("keep_messages cannot be negative.")
        thread = self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(thread["summary"])
        total_tokens = estimate_tokens(summary) + sum(
            estimate_tokens(item["content"]) for item in messages
        )
        if total_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = len(messages) - self.keep_messages
        compacted = messages[:split_at]
        thread["summary"] = summarize_messages(
            ([{"role": "summary", "content": summary}] if summary else []) + compacted,
            max_items=6,
        )
        thread["messages"] = messages[split_at:]
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}
        return {
            "messages": [dict(message) for message in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
