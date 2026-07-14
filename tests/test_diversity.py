from learnbridge.diversity import mmr_rerank
from learnbridge.preprocessing import resource_text


def test_provider_limit(resources, bundle):
    candidates = resources.head(20).copy()
    candidates["provider"] = "One Learning"
    candidates.loc[2:, "provider"] = "Two Learning"
    candidates["base_score"] = .8
    vectors = bundle.vectorizer.transform(candidates.apply(resource_text, axis=1))
    result = mmr_rerank(candidates, vectors, 5, .7, 2)
    assert result.provider.value_counts().max() <= 2


def test_mmr_returns_unique_items(resources, bundle):
    candidates = resources.head(20).copy()
    candidates["base_score"] = .8
    vectors = bundle.vectorizer.transform(candidates.apply(resource_text, axis=1))
    result = mmr_rerank(candidates, vectors, 5)
    assert result.resource_id.is_unique and len(result) == 5
