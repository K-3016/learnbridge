"""Typed input and output schemas."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearnerProfile:
    learning_goal: str
    current_level: str = "Beginner"
    weak_topics: list[str] = field(default_factory=list)
    preferred_formats: list[str] = field(default_factory=list)
    weekly_time_minutes: int = 180
    free_only: bool = False
    requires_captions: bool = False
    exploration_level: float = 0.3
    profile_id: str = "interactive"

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "LearnerProfile":
        def split(value: Any) -> list[str]:
            if isinstance(value, list):
                return value
            return [x.strip() for x in str(value or "").split("|") if x.strip()]
        return cls(
            profile_id=str(row.get("profile_id", "interactive")),
            learning_goal=str(row.get("learning_goal", "")),
            current_level=str(row.get("current_level", "Beginner")),
            weak_topics=split(row.get("weak_topics")),
            preferred_formats=split(row.get("preferred_formats")),
            weekly_time_minutes=int(row.get("weekly_time_minutes", 180)),
            free_only=str(row.get("free_only", False)).lower() in {"true", "1"},
            requires_captions=str(row.get("requires_captions", False)).lower() in {"true", "1"},
            exploration_level=float(row.get("exploration_level", 0.3)),
        )


@dataclass
class Recommendation:
    resource_id: str
    rank: int
    title: str
    match_score: float
    responsible_score: float
    reason: str
    knowledge_gap_addressed: str
    responsible_note: str
    metadata: dict[str, Any]
