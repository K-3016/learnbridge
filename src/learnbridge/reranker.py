"""Responsible hybrid scoring and reranking."""
import pandas as pd
from .config import MAX_PROVIDER_TOP5, SCORE_WEIGHTS
from .constraints import time_fit_score
from .diversity import mmr_rerank
from .preprocessing import resource_text
from .schemas import LearnerProfile


def _gap_score(topics: str, weak_topics: list[str]) -> float:
    resource_topics = {x.strip().lower() for x in str(topics).split("|")}
    weak = {x.lower() for x in weak_topics}
    return len(resource_topics & weak) / max(1, len(weak))


def responsible_rerank(candidates: pd.DataFrame, profile: LearnerProfile, vectorizer, top_k: int = 5, mmr_lambda: float = 0.7) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    scored = candidates.copy()
    scored["knowledge_gap_score"] = scored.topics.map(lambda x: _gap_score(x, profile.weak_topics))
    scored["quality_component"] = scored.quality_score.clip(0, 1)
    scored["accessibility_component"] = scored.apply(
        lambda r: min(1.0, float(r.accessibility_score) * (1.0 if time_fit_score(r.duration_minutes, profile.weekly_time_minutes) == 1 else 0.8)), axis=1
    )
    w = SCORE_WEIGHTS
    # Diversity is applied by MMR, then included explicitly in the reported final score.
    scored["base_score"] = (
        w["relevance"] * scored.relevance_score.clip(0, 1)
        + w["knowledge_gap"] * scored.knowledge_gap_score
        + w["quality"] * scored.quality_component
        + w["accessibility"] * scored.accessibility_component
    ) / (1 - w["diversity"])
    vectors = vectorizer.transform(scored.apply(resource_text, axis=1))
    result = mmr_rerank(scored, vectors, top_k, mmr_lambda, MAX_PROVIDER_TOP5)
    result["responsible_score"] = (
        w["relevance"] * result.relevance_score.clip(0, 1)
        + w["knowledge_gap"] * result.knowledge_gap_score
        + w["quality"] * result.quality_component
        + w["diversity"] * result.diversity_contribution
        + w["accessibility"] * result.accessibility_component
    )
    return result.sort_values("mmr_score", ascending=False).reset_index(drop=True)
