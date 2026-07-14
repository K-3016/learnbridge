"""Central project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = ROOT / "models"
RESOURCE_FILE = PROCESSED_DIR / "resources.csv"
PROFILE_FILE = PROCESSED_DIR / "profiles.csv"
JUDGMENT_FILE = PROCESSED_DIR / "judgments.csv"
FEATURE_FILE = MODEL_DIR / "tfidf_features.joblib"
SEED = 42
TOP_K = 5
CANDIDATE_K = 30
MMR_LAMBDA = 0.7
MAX_PROVIDER_TOP5 = 2
SCORE_WEIGHTS = {
    "relevance": 0.60,
    "knowledge_gap": 0.15,
    "quality": 0.10,
    "diversity": 0.10,
    "accessibility": 0.05,
}

TOPICS = [
    "Python", "Data science", "Machine learning", "Statistics", "SQL", "NLP",
    "Computer vision", "Deep learning", "Cloud computing", "Software engineering",
    "Responsible AI", "Mathematics",
]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
FORMATS = ["Video", "Article", "Interactive exercise", "Course", "Tutorial", "Practice quiz", "Project"]
