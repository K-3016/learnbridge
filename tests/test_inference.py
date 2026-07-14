from learnbridge.app_utils import explanation_mode
from learnbridge.inference import recommend


def test_end_to_end(resources, bundle, profile):
    recs, _ = recommend(resources, profile, bundle, "responsible", 5)
    assert len(recs) == 5
    assert {"reason", "responsible_note", "match_score"}.issubset(recs.columns)


def test_all_models(resources, bundle, profile):
    for model in ["popularity", "content", "responsible"]:
        assert len(recommend(resources, profile, bundle, model, 3)[0]) == 3


def test_llm_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("LEARNBRIDGE_ENABLE_LLM", raising=False)
    assert "fallback" in explanation_mode(tmp_path / "absent")


def test_unknown_model(resources, bundle, profile):
    import pytest
    with pytest.raises(ValueError):
        recommend(resources, profile, bundle, "unknown")
