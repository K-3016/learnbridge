import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.make_dataset import generate_resources
from learnbridge.retrieval import build_feature_bundle
from learnbridge.schemas import LearnerProfile


@pytest.fixture
def resources():
    return generate_resources(40, 42)


@pytest.fixture
def bundle(resources):
    return build_feature_bundle(resources)


@pytest.fixture
def profile():
    return LearnerProfile("learn machine learning with Python", "Beginner", ["Statistics", "Machine learning"], ["Video", "Tutorial"], 180, True, True, .4)
