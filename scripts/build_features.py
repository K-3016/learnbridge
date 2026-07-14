"""Build and persist TF-IDF retrieval artifacts."""
import argparse
import joblib
from learnbridge.config import FEATURE_FILE
from learnbridge.data_loader import load_resources
from learnbridge.retrieval import build_feature_bundle


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=str(FEATURE_FILE)); args = parser.parse_args()
    bundle = build_feature_bundle(load_resources())
    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)


if __name__ == "__main__": main()
