"""Generate deterministic, original synthetic prototype data."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

from learnbridge.config import DIFFICULTIES, FORMATS, PROCESSED_DIR, RAW_DIR, SEED, TOPICS

PROVIDERS = [f"{a} Learning" for a in ["Bridge", "Orbit", "Cedar", "Nova", "Atlas", "Lumen", "Sprout", "Pixel", "Summit", "Harbor", "Quill", "Mosaic", "Circuit", "Willow", "Beacon", "Forge", "Compass", "Nexus"]]
GOALS = [
    "build practical machine learning systems", "learn data analysis with Python", "understand responsible AI",
    "prepare for cloud data projects", "strengthen statistics and mathematics", "create NLP applications",
    "improve SQL and data science skills", "learn computer vision and deep learning", "write reliable software",
]


def generate_resources(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    verbs = ["Foundations", "Workshop", "Field Guide", "Practice Lab", "Essentials", "Applied Path", "Concept Clinic"]
    for i in range(n):
        primary = rng.choice(TOPICS)
        secondary = rng.choice([x for x in TOPICS if x != primary])
        difficulty = rng.choices(DIFFICULTIES, [0.42, 0.38, 0.20])[0]
        fmt = rng.choice(FORMATS)
        cost = rng.choices(["Free", "Paid"], [0.68, 0.32])[0]
        duration = rng.choice([20, 30, 45, 60, 90, 120, 180, 240, 360, 480])
        captions = fmt != "Video" or rng.random() < 0.78
        transcript = fmt in {"Article", "Tutorial"} or captions or rng.random() < 0.35
        accessibility = min(1.0, 0.35 + 0.25 * captions + 0.2 * transcript + rng.random() * 0.2)
        prereq = "None" if difficulty == "Beginner" else rng.choice([primary, "Python", "Mathematics", "Statistics"])
        title = f"{primary} {rng.choice(verbs)} {i + 1}"
        desc = f"An original {difficulty.lower()} learning resource that develops {primary.lower()} concepts through {fmt.lower()} activities and connects them with {secondary.lower()} practice."
        rows.append({
            "resource_id": f"R{i+1:04d}", "title": title, "description": desc,
            "topics": f"{primary}|{secondary}", "difficulty": difficulty, "format": fmt,
            "duration_minutes": duration, "cost_type": cost, "price_usd": 0.0 if cost == "Free" else float(rng.choice([9, 15, 25, 39, 59])),
            "provider": rng.choice(PROVIDERS), "quality_score": round(rng.uniform(0.45, 0.99), 3),
            "accessibility_score": round(accessibility, 3), "has_captions": captions, "has_transcript": transcript,
            "prerequisites": prereq, "language": "English", "resource_url": f"https://example.org/learnbridge/resources/R{i+1:04d}",
            "popularity_score": round(rng.betavariate(2.2, 3.0), 3),
        })
    return pd.DataFrame(rows)


def generate_profiles(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    rows = []
    for i in range(n):
        weak = rng.sample(TOPICS, rng.choice([1, 2, 3]))
        rows.append({
            "profile_id": f"P{i+1:03d}", "learning_goal": rng.choice(GOALS), "current_level": rng.choice(DIFFICULTIES),
            "weak_topics": "|".join(weak), "preferred_formats": "|".join(rng.sample(FORMATS, 2)),
            "weekly_time_minutes": rng.choice([60, 90, 120, 180, 240, 360]), "free_only": rng.random() < 0.45,
            "requires_captions": rng.random() < 0.25, "exploration_level": round(rng.uniform(0, 1), 2),
        })
    return pd.DataFrame(rows)


def generate_judgments(resources: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in profiles.iterrows():
        weak = set(p.weak_topics.split("|")); goal = set(str(p.learning_goal).lower().split())
        for _, r in resources.iterrows():
            topics = set(r.topics.split("|")); overlap = bool(weak & topics)
            goal_overlap = bool(goal & set((r.title + " " + r.description).lower().split()))
            compatible = not (p.current_level == "Beginner" and r.difficulty == "Advanced")
            constraint_ok = (not p.free_only or r.cost_type == "Free") and (not p.requires_captions or r["format"] != "Video" or r.has_captions)
            rel = 2 if overlap and compatible and constraint_ok else 1 if (overlap or goal_overlap) and constraint_ok else 0
            rows.append({"profile_id": p.profile_id, "resource_id": r.resource_id, "relevance": rel})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--resources", type=int, default=600); parser.add_argument("--profiles", type=int, default=72); parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.resources < 500 or args.profiles < 60:
        raise ValueError("Full dataset requires at least 500 resources and 60 profiles")
    RAW_DIR.mkdir(parents=True, exist_ok=True); PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    resources = generate_resources(args.resources, args.seed); profiles = generate_profiles(args.profiles, args.seed); judgments = generate_judgments(resources, profiles)
    resources.to_json(RAW_DIR / "resources.jsonl", orient="records", lines=True); profiles.to_json(RAW_DIR / "profiles.jsonl", orient="records", lines=True)
    resources.to_csv(PROCESSED_DIR / "resources.csv", index=False); profiles.to_csv(PROCESSED_DIR / "profiles.csv", index=False); judgments.to_csv(PROCESSED_DIR / "judgments.csv", index=False)
    metadata = {"synthetic": True, "purpose": "prototype evaluation only", "seed": args.seed, "resources": len(resources), "profiles": len(profiles)}
    (PROCESSED_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
