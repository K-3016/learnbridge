from learnbridge.constraints import apply_constraints


def test_free_only_filter(resources, profile):
    result = apply_constraints(resources, profile)
    assert (result.eligible.cost_type == "Free").all()


def test_caption_filter(resources, profile):
    result = apply_constraints(resources, profile)
    videos = result.eligible[result.eligible["format"] == "Video"]
    assert videos.has_captions.all()


def test_beginner_advanced_filter(resources, profile):
    profile.exploration_level = .2
    result = apply_constraints(resources, profile)
    assert not (result.eligible.difficulty == "Advanced").any()


def test_rejection_reasons(resources, profile):
    assert apply_constraints(resources, profile).rejected.filter_reason.str.len().gt(0).all()
