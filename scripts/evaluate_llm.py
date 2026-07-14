"""Evaluate real base/adapted outputs without fabricating absent results."""
import argparse
import json
from pathlib import Path


def score_record(record):
    text=json.dumps(record.get("output",{})).lower(); allowed=set(record.get("allowed_resource_ids",[])); ranked=set(record.get("output",{}).get("ranking",[]))
    rejected=set(record.get("must_reject_ids",[])); claims=record.get("output",{}).get("claims",[])
    return {"constraint_following":float(not (ranked&rejected)),"groundedness":float(ranked.issubset(allowed)),"ranking_relevance":float(record.get("ranking_relevance",0)),"unsupported_claim_rate":sum(not c.get("supported",False) for c in claims)/max(1,len(claims))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-results"); p.add_argument("--adapted-results"); p.add_argument("--output",default="data/outputs/llm_comparison.json"); args=p.parse_args()
    payload={"computed":False,"message":"No model outputs supplied. Train/run both models, then pass JSONL result files."}
    if args.base_results and args.adapted_results:
        payload={"computed":True,"models":{}}
        for name,path in [("base",args.base_results),("adapted",args.adapted_results)]:
            records=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]; scores=[score_record(x) for x in records]
            payload["models"][name]={k:sum(s[k] for s in scores)/len(scores) for k in scores[0]} if scores else {}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2))


if __name__=="__main__": main()
