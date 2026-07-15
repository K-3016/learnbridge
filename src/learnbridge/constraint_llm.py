"""Balanced data and metrics for the LLM constraint-classification experiment."""
from __future__ import annotations

import random
from typing import Any

import pandas as pd

LABELS = ["ACCEPT", "REJECT_BUDGET", "REJECT_CAPTIONS", "REJECT_LEVEL"]


def decision_prompt(profile: dict[str, Any], resource: dict[str, Any]) -> str:
    """Create a short, unambiguous constraint-decision prompt."""
    return (
        "Decide whether this learning resource satisfies the learner's hard constraints.\n"
        f"Learner: level={profile['level']}; free_only={profile['free_only']}; "
        f"captions_required={profile['captions_required']}\n"
        f"Resource: id={resource['resource_id']}; difficulty={resource['difficulty']}; "
        f"format={resource['format']}; cost={resource['cost_type']}; "
        f"captions={bool(resource['has_captions'])}\n"
        "Reply with exactly one label: ACCEPT, REJECT_BUDGET, REJECT_CAPTIONS, "
        "or REJECT_LEVEL.\nDecision:"
    )


def _profile_for(label: str) -> dict[str, Any]:
    profiles = {
        "ACCEPT": {"level": "Beginner", "free_only": True, "captions_required": True},
        "REJECT_BUDGET": {
            "level": "Beginner",
            "free_only": True,
            "captions_required": False,
        },
        "REJECT_CAPTIONS": {
            "level": "Beginner",
            "free_only": False,
            "captions_required": True,
        },
        "REJECT_LEVEL": {
            "level": "Beginner",
            "free_only": False,
            "captions_required": False,
        },
    }
    return profiles[label]


def _controlled_resource(row: dict[str, Any], label: str) -> dict[str, Any]:
    """Set only constraint fields to create an unambiguous synthetic case."""
    resource = dict(row)
    resource.update(
        {
            "difficulty": "Beginner",
            "format": "Tutorial",
            "cost_type": "Free",
            "has_captions": True,
        }
    )
    if label == "REJECT_BUDGET":
        resource["cost_type"] = "Paid"
    elif label == "REJECT_CAPTIONS":
        resource["format"] = "Video"
        resource["has_captions"] = False
    elif label == "REJECT_LEVEL":
        resource["difficulty"] = "Advanced"
    return resource


def build_balanced_examples(
    resources: pd.DataFrame,
    per_label: int,
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build balanced labeled examples from real synthetic catalog rows."""
    excluded = excluded_ids or set()
    examples: list[dict[str, Any]] = []
    used: set[str] = set()
    for label_index, label in enumerate(LABELS):
        pool = resources
        pool = pool[~pool.resource_id.astype(str).isin(excluded | used)]
        if len(pool) < per_label:
            raise ValueError(f"Not enough unique resources for {label}: {len(pool)}")
        sampled = pool.sample(per_label, random_state=seed + label_index)
        profile = _profile_for(label)
        for _, row in sampled.iterrows():
            resource = _controlled_resource(row.to_dict(), label)
            resource_id = str(resource["resource_id"])
            used.add(resource_id)
            examples.append(
                {
                    "resource_id": resource_id,
                    "prompt": decision_prompt(profile, resource),
                    "completion": label,
                    "label": label,
                }
            )
    random.Random(seed).shuffle(examples)
    return examples


def parse_label(text: str) -> str | None:
    """Parse one of the permitted labels from generated text."""
    normalized = text.strip().upper()
    for label in sorted(LABELS, key=len, reverse=True):
        if normalized.startswith(label):
            return label
    return None
