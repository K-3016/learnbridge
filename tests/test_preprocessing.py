from scripts.make_dataset import generate_resources
from learnbridge.data_loader import RESOURCE_COLUMNS
from learnbridge.preprocessing import clean_text, profile_text


def test_schema(resources):
    assert RESOURCE_COLUMNS.issubset(resources.columns)


def test_deterministic_generation():
    assert generate_resources(8, 9).equals(generate_resources(8, 9))


def test_clean_text():
    assert clean_text("  NLP, SQL!! ") == "nlp sql"


def test_profile_text(profile):
    text = profile_text(profile)
    assert "machine learning" in text and text.count("statistics") == 2
