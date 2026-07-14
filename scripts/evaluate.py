"""Compute all offline metrics and trade-off results from generated data."""
import argparse
import json
from collections import defaultdict
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from learnbridge.config import FEATURE_FILE, FIGURE_DIR, OUTPUT_DIR
from learnbridge.data_loader import load_judgments, load_profiles, load_resources
from learnbridge.inference import recommend
from learnbridge.metrics import (categorical_diversity, constraint_satisfaction, intra_list_diversity,
    knowledge_gap_coverage, ndcg_at_k, precision_at_k, recall_at_k, topic_diversity)
from learnbridge.preprocessing import resource_text
from learnbridge.schemas import LearnerProfile


def evaluate_model(name, resources, profiles, judgments, bundle, lambda_=None):
    rows=[]; all_ids=set(); provider_counts=defaultdict(int)
    for _, p_row in profiles.iterrows():
        p=LearnerProfile.from_mapping(p_row.to_dict()); recs,_=recommend(resources,p,bundle,name,5,lambda_)
        rel_map=judgments[judgments.profile_id==p.profile_id].set_index("resource_id").relevance.to_dict()
        rels=[int(rel_map.get(x,0)) for x in recs.resource_id]; all_rels=list(rel_map.values()); relevant=sum(x>0 for x in all_rels)
        vec=bundle.vectorizer.transform(recs.apply(resource_text,axis=1))
        for provider in recs.provider: provider_counts[str(provider)]+=1
        rows.append({"model":name if lambda_ is None else f"responsible λ={lambda_}","profile_id":p.profile_id,
            "precision_at_5":precision_at_k(rels),"recall_at_5":recall_at_k(rels,relevant),"ndcg_at_5":ndcg_at_k(rels,sorted(all_rels,reverse=True)),
            "mean_relevance_at_5":sum(rels)/max(1,len(rels)),"intra_list_diversity_at_5":intra_list_diversity(vec),
            "topic_diversity_at_5":topic_diversity(recs),"format_diversity_at_5":categorical_diversity(recs["format"].tolist()),
            "provider_diversity_at_5":categorical_diversity(recs.provider.tolist()),"constraint_satisfaction_rate":constraint_satisfaction(recs,p),
            "knowledge_gap_coverage":knowledge_gap_coverage(recs,p.weak_topics),"average_popularity":recs.popularity_score.mean(),
            "long_tail_exposure":float((recs.popularity_score<resources.popularity_score.quantile(.5)).mean())})
        all_ids.update(recs.resource_id)
    frame=pd.DataFrame(rows); frame["catalog_coverage"]=len(all_ids)/len(resources)
    return frame,provider_counts


def save_plots(summary, tradeoff, provider):
    plt.style.use("seaborn-v0_8-whitegrid"); FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    primary=["ndcg_at_5","intra_list_diversity_at_5","constraint_satisfaction_rate","catalog_coverage"]
    ax=summary.set_index("model")[primary].plot(kind="bar",figsize=(10,5)); ax.set(title="LearnBridge Model Comparison",ylabel="Metric value (0–1)",xlabel="Model"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIGURE_DIR/"model_comparison.png",dpi=160); plt.close()
    ax=tradeoff.plot(x="intra_list_diversity_at_5",y="ndcg_at_5",marker="o",figsize=(7,5),legend=False)
    for _,r in tradeoff.iterrows(): ax.annotate(r["model"],(r.intra_list_diversity_at_5,r.ndcg_at_5))
    ax.set(title="Relevance–Diversity Trade-off",xlabel="Intra-list diversity@5",ylabel="NDCG@5"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"tradeoff.png",dpi=160); plt.close()
    for metric,file,title in [("constraint_satisfaction_rate","constraint_satisfaction.png","Constraint Satisfaction Comparison"),("catalog_coverage","catalog_coverage.png","Catalog Coverage Comparison")]:
        ax=summary.plot.bar(x="model",y=metric,legend=False,figsize=(7,4)); ax.bar_label(ax.containers[0],fmt="%.2f"); ax.set(title=title,ylabel=metric.replace("_"," ").title(),xlabel="Model",ylim=(0,1.05)); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIGURE_DIR/file,dpi=160); plt.close()
    pf=pd.DataFrame(provider.items(),columns=["provider","exposure"]).sort_values("exposure",ascending=False)
    ax=pf.plot.bar(x="provider",y="exposure",legend=False,figsize=(11,5)); ax.set(title="Responsible Model Provider Exposure",ylabel="Top-5 appearances",xlabel="Provider"); plt.xticks(rotation=55,ha="right"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"provider_exposure.png",dpi=160); plt.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--profile-limit",type=int); args=parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); resources=load_resources(); profiles=load_profiles(); judgments=load_judgments(); bundle=joblib.load(FEATURE_FILE)
    if args.profile_limit: profiles=profiles.head(args.profile_limit)
    frames=[]; providers={}
    for name in ["popularity","content","responsible"]:
        frame,p=evaluate_model(name,resources,profiles,judgments,bundle); frames.append(frame); providers[name]=p
    detailed=pd.concat(frames,ignore_index=True); metrics=[c for c in detailed if c not in {"model","profile_id"}]
    summary=detailed.groupby("model")[metrics].mean().reset_index(); summary.to_csv(OUTPUT_DIR/"model_comparison.csv",index=False); detailed.to_csv(OUTPUT_DIR/"per_profile_metrics.csv",index=False)
    trade=[]
    for value in [1.0,.9,.7,.5,.3]:
        frame,_=evaluate_model("responsible",resources,profiles,judgments,bundle,value); trade.append(frame)
    trade_detail=pd.concat(trade,ignore_index=True); trade_summary=trade_detail.groupby("model")[metrics].mean().reset_index(); trade_summary.to_csv(OUTPUT_DIR/"tradeoff_results.csv",index=False)
    pd.DataFrame(providers["responsible"].items(),columns=["provider","exposure"]).to_csv(OUTPUT_DIR/"provider_exposure.csv",index=False)
    summary_dict={r["model"]:{k:round(float(r[k]),4) for k in metrics} for _,r in summary.iterrows()}
    (OUTPUT_DIR/"evaluation_summary.json").write_text(json.dumps({"computed":True,"synthetic_data":True,"models":summary_dict},indent=2))
    save_plots(summary,trade_summary,providers["responsible"])


if __name__=="__main__": main()
