"""Data loading and validation helpers."""
from pathlib import Path
import pandas as pd

from .config import JUDGMENT_FILE, PROFILE_FILE, RESOURCE_FILE

RESOURCE_COLUMNS = {
    "resource_id", "title", "description", "topics", "difficulty", "format",
    "duration_minutes", "cost_type", "price_usd", "provider", "quality_score",
    "accessibility_score", "has_captions", "has_transcript", "prerequisites",
    "language", "resource_url", "popularity_score",
}


def load_resources(path: Path = RESOURCE_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Resource data missing at {path}. Run: make data")
    frame = pd.read_csv(path)
    missing = RESOURCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Resource data is missing columns: {sorted(missing)}")
    for column in ("has_captions", "has_transcript"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    return frame


def load_profiles(path: Path = PROFILE_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Profile data missing at {path}. Run: make data")
    return pd.read_csv(path)


def load_judgments(path: Path = JUDGMENT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Judgments missing at {path}. Run: make data")
    return pd.read_csv(path)
