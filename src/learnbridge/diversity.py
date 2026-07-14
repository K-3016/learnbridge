"""Greedy MMR diversification with provider exposure control."""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def mmr_rerank(candidates: pd.DataFrame, vectors, top_k: int = 5, lambda_: float = 0.7, max_per_provider: int = 2) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    providers: Counter[str] = Counter()
    base = candidates["base_score"].to_numpy(float)
    while remaining and len(selected) < top_k:
        choices = []
        for i in remaining:
            provider = str(candidates.iloc[i].provider)
            if providers[provider] >= max_per_provider:
                continue
            similarity = 0.0 if not selected else float(cosine_similarity(vectors[i], vectors[selected]).max())
            # Exploration lowers lambda outside this function; novelty is recorded for transparency.
            mmr = lambda_ * base[i] - (1 - lambda_) * similarity
            choices.append((mmr, -i, i, similarity))
        if not choices:
            break
        mmr, _, chosen, similarity = max(choices)
        selected.append(chosen)
        remaining.remove(chosen)
        providers[str(candidates.iloc[chosen].provider)] += 1
    result = candidates.iloc[selected].copy()
    result["diversity_contribution"] = [1.0 if j == 0 else 1.0 - float(cosine_similarity(vectors[i], vectors[selected[:j]]).max()) for j, i in enumerate(selected)]
    result["mmr_score"] = [lambda_ * base[i] - (1 - lambda_) * (1 - d) for i, d in zip(selected, result.diversity_contribution)]
    return result.reset_index(drop=True)
