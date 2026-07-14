"""Grounded deterministic recommendation explanations."""
from pathlib import Path
import os
import pandas as pd
from .schemas import LearnerProfile


FORBIDDEN_CLAIMS = {"certificate", "accredited", "salary", "guaranteed job", "instructor credential"}


def explanation_mode(adapter_dir: Path | None = None) -> str:
    enabled = os.getenv("LEARNBRIDGE_ENABLE_LLM", "0") == "1"
    return "optional LoRA adapter" if enabled and adapter_dir and adapter_dir.exists() else "deterministic metadata fallback"


def explain(row: pd.Series, profile: LearnerProfile) -> tuple[str, str, str]:
    topics = [x.strip() for x in str(row.topics).split("|")]
    weak = [x for x in topics if x.lower() in {w.lower() for w in profile.weak_topics}]
    gap = ", ".join(weak) if weak else "your stated learning goal"
    reasons = [f"covers {', '.join(topics[:2])}", f"matches the {row.difficulty.lower()} level"]
    if row["format"] in profile.preferred_formats:
        reasons.append(f"uses your preferred {row['format'].lower()} format")
    if float(row.duration_minutes) <= profile.weekly_time_minutes:
        reasons.append("fits within your weekly study time")
    reason = "Recommended because it " + ", ".join(reasons) + "."
    note_parts = ["Selected after budget, accessibility, difficulty, and provider checks"]
    if weak:
        note_parts.append("addresses a declared knowledge gap")
    note_parts.append("diversified against the rest of the list")
    return reason, gap, "; ".join(note_parts) + "."
