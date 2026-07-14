"""Popularity and TF-IDF candidate retrieval."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import profile_text, resource_text
from .schemas import LearnerProfile


@dataclass
class FeatureBundle:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    resource_ids: list[str]


def build_feature_bundle(resources: pd.DataFrame) -> FeatureBundle:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(resources.apply(resource_text, axis=1))
    return FeatureBundle(vectorizer, matrix.tocsr(), resources.resource_id.astype(str).tolist())


def popularity_retrieve(resources: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    result = resources.copy()
    result["relevance_score"] = 0.55 * result["quality_score"] + 0.45 * result["popularity_score"]
    return result.sort_values("relevance_score", ascending=False).head(k).reset_index(drop=True)


def content_retrieve(resources: pd.DataFrame, profile: LearnerProfile, bundle: FeatureBundle, k: int = 20) -> pd.DataFrame:
    query = bundle.vectorizer.transform([profile_text(profile)])
    scores = cosine_similarity(query, bundle.matrix).ravel()
    indices = np.argsort(-scores)[: min(k, len(resources))]
    result = resources.iloc[indices].copy()
    result["relevance_score"] = scores[indices]
    return result.reset_index(drop=True)
