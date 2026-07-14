"""Text preprocessing for resources and cold-start profiles."""
import re
import pandas as pd
from .schemas import LearnerProfile


def clean_text(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9+#. ]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def resource_text(row: pd.Series) -> str:
    fields = [row.get(x, "") for x in ("title", "description", "topics", "difficulty", "format", "prerequisites")]
    return clean_text(" ".join(map(str, fields)))


def profile_text(profile: LearnerProfile) -> str:
    # Repeat weak topics to make gap coverage meaningful in simple TF-IDF retrieval.
    parts = [profile.learning_goal, profile.current_level, " ".join(profile.preferred_formats)]
    parts.extend(profile.weak_topics * 2)
    return clean_text(" ".join(parts))
