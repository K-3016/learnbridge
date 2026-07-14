"""Generate example lists and a five-minute pitch asset summary."""
import argparse
import json
import joblib
from learnbridge.config import FEATURE_FILE, OUTPUT_DIR
from learnbridge.data_loader import load_profiles, load_resources
from learnbridge.inference import recommend
from learnbridge.schemas import LearnerProfile


def main():
    argparse.ArgumentParser().parse_args(); resources=load_resources(); bundle=joblib.load(FEATURE_FILE); profile=LearnerProfile.from_mapping(load_profiles().iloc[0].to_dict())
    payload={"profile":profile.__dict__}
    for model in ["popularity","content","responsible"]:
        recs,_=recommend(resources,profile,bundle,model); payload[model]=recs[["rank","resource_id","title","provider","match_score","responsible_score"]].to_dict("records")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); (OUTPUT_DIR/"example_before_after.json").write_text(json.dumps(payload,indent=2))


if __name__=="__main__": main()
