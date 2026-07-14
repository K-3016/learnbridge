"""Hard constraints and compatibility preferences."""
from dataclasses import dataclass
import pandas as pd
from .schemas import LearnerProfile


@dataclass
class ConstraintResult:
    eligible: pd.DataFrame
    rejected: pd.DataFrame


def apply_constraints(candidates: pd.DataFrame, profile: LearnerProfile) -> ConstraintResult:
    keep, rejected = [], []
    weak = {x.lower() for x in profile.weak_topics}
    for idx, row in candidates.iterrows():
        reasons: list[str] = []
        if profile.free_only and str(row.cost_type).lower() != "free":
            reasons.append("excluded by free-only budget constraint")
        if profile.requires_captions and row["format"] == "Video" and not bool(row.has_captions):
            reasons.append("video lacks required captions")
        prereqs = {x.strip().lower() for x in str(row.prerequisites).split("|") if x.strip() and x != "None"}
        exploratory_ok = profile.exploration_level >= 0.7 and prereqs.issubset(weak | {profile.current_level.lower()})
        if profile.current_level == "Beginner" and row.difficulty == "Advanced" and not exploratory_ok:
            reasons.append("advanced difficulty is incompatible with beginner level")
        target = rejected if reasons else keep
        item = row.to_dict()
        item["filter_reason"] = "; ".join(reasons)
        target.append(item)
    return ConstraintResult(pd.DataFrame(keep, columns=list(candidates.columns) + ["filter_reason"]), pd.DataFrame(rejected))


def time_fit_score(duration: float, weekly_minutes: int) -> float:
    if duration <= weekly_minutes:
        return 1.0
    return max(0.0, weekly_minutes / max(duration, 1.0))
