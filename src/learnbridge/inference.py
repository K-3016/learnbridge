"""End-to-end recommendation service used by CLI and Streamlit."""
from dataclasses import asdict
import pandas as pd
from .config import CANDIDATE_K, MMR_LAMBDA
from .constraints import apply_constraints
from .retrieval import FeatureBundle, content_retrieve, popularity_retrieve
from .reranker import responsible_rerank
from .schemas import LearnerProfile
from .app_utils import explain


def _decorate(frame: pd.DataFrame, profile: LearnerProfile, responsible: bool = False) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    if "responsible_score" not in result:
        result["responsible_score"] = result["relevance_score"]
    explanations = result.apply(lambda row: explain(row, profile), axis=1)
    result["reason"] = [x[0] for x in explanations]
    result["knowledge_gap_addressed"] = [x[1] for x in explanations]
    result["responsible_note"] = [x[2] for x in explanations]
    result["rank"] = range(1, len(result) + 1)
    result["match_score"] = result["relevance_score"].clip(0, 1)
    return result


def recommend(resources: pd.DataFrame, profile: LearnerProfile, bundle: FeatureBundle, model: str = "responsible", top_k: int = 5, mmr_lambda: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model == "popularity":
        candidates = popularity_retrieve(resources, max(CANDIDATE_K, top_k))
        constrained = apply_constraints(candidates, profile)
        return _decorate(constrained.eligible.head(top_k), profile), constrained.rejected
    candidates = content_retrieve(resources, profile, bundle, max(CANDIDATE_K, top_k))
    constrained = apply_constraints(candidates, profile)
    if model == "content":
        return _decorate(constrained.eligible.head(top_k), profile), constrained.rejected
    if model != "responsible":
        raise ValueError(f"Unknown model: {model}")
    # Map a 0..1 exploration control onto relevance-heavy to diversity-heavy MMR.
    effective_lambda = mmr_lambda if mmr_lambda is not None else max(0.3, min(1.0, 1.0 - 0.7 * profile.exploration_level))
    ranked = responsible_rerank(constrained.eligible, profile, bundle.vectorizer, top_k, effective_lambda)
    return _decorate(ranked, profile, True), constrained.rejected
