from learnbridge.retrieval import content_retrieve, popularity_retrieve


def test_tfidf_retrieval_count(resources, bundle, profile):
    assert len(content_retrieve(resources, profile, bundle, 20)) == 20


def test_retrieval_is_sorted(resources, bundle, profile):
    result = content_retrieve(resources, profile, bundle, 10)
    assert result.relevance_score.is_monotonic_decreasing


def test_popularity_is_profile_independent(resources):
    assert len(popularity_retrieve(resources, 5)) == 5
