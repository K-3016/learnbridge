"""Validate the deterministic recommendation model artifact."""
import argparse
import joblib
from learnbridge.config import FEATURE_FILE
from learnbridge.data_loader import load_profiles, load_resources
from learnbridge.inference import recommend
from learnbridge.schemas import LearnerProfile


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--top-k", type=int, default=5); args = parser.parse_args()
    bundle = joblib.load(FEATURE_FILE); resources = load_resources(); profile = LearnerProfile.from_mapping(load_profiles().iloc[0].to_dict())
    results, _ = recommend(resources, profile, bundle, "responsible", args.top_k)
    if len(results) < args.top_k: raise RuntimeError("Model could not produce the requested recommendations")


if __name__ == "__main__": main()
