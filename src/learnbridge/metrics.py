"""Offline relevance and responsible recommendation metrics."""
from collections import Counter
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .constraints import apply_constraints
from .schemas import LearnerProfile


def precision_at_k(relevances: list[int], k: int = 5) -> float:
    values = relevances[:k]
    return sum(x > 0 for x in values) / k


def recall_at_k(relevances: list[int], total_relevant: int, k: int = 5) -> float:
    return sum(x > 0 for x in relevances[:k]) / max(1, total_relevant)


def ndcg_at_k(relevances: list[int], ideal: list[int] | None = None, k: int = 5) -> float:
    def dcg(vals: list[int]) -> float:
        return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(vals[:k]))
    ideal_values = sorted(ideal if ideal is not None else relevances, reverse=True)
    denom = dcg(ideal_values)
    return dcg(relevances) / denom if denom else 0.0


def categorical_diversity(values: list[str]) -> float:
    return len(set(values)) / max(1, len(values))


def topic_diversity(items: pd.DataFrame) -> float:
    topics = [t.strip() for value in items.topics for t in str(value).split("|")]
    return len(set(topics)) / max(1, len(topics))


def intra_list_diversity(vectors) -> float:
    if vectors.shape[0] < 2:
        return 0.0
    similarities = cosine_similarity(vectors)
    upper = similarities[np.triu_indices_from(similarities, k=1)]
    return float(1 - upper.mean())


def constraint_satisfaction(items: pd.DataFrame, profile: LearnerProfile) -> float:
    if items.empty:
        return 0.0
    return len(apply_constraints(items, profile).eligible) / len(items)


def knowledge_gap_coverage(items: pd.DataFrame, weak_topics: list[str]) -> float:
    present = {x.strip().lower() for value in items.topics for x in str(value).split("|")}
    weak = {x.lower() for x in weak_topics}
    return len(present & weak) / max(1, len(weak))


def provider_distribution(lists: list[pd.DataFrame]) -> dict[str, int]:
    return dict(Counter(str(p) for frame in lists for p in frame.provider))
