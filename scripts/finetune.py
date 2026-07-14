"""Optional LoRA instruction tuning; dry-run never downloads a model."""
import argparse
import json
import random
from pathlib import Path
from learnbridge.data_loader import load_profiles, load_resources


def build_examples(limit:int,seed:int):
    rng=random.Random(seed); resources=load_resources(); profiles=load_profiles(); examples=[]
    for _,p in profiles.iterrows():
        candidates=resources.sample(6,random_state=seed+len(examples)).to_dict("records")
        free=[r for r in candidates if not p.free_only or r["cost_type"]=="Free"]
        prompt={"learner_profile":p.to_dict(),"candidate_resources":[{k:r[k] for k in ["resource_id","title","topics","difficulty","format","duration_minutes","cost_type","provider","has_captions"]} for r in candidates],"constraints":["budget","captions","difficulty","ground explanations in metadata"]}
        response={"ranking":[r["resource_id"] for r in free[:5]],"explanations":[f"{r['title']} covers {r['topics']} in {r['format']} format." for r in free[:5]],"rejected":[{"resource_id":r["resource_id"],"reason":"violates free-only constraint"} for r in candidates if p.free_only and r["cost_type"]!="Free"]}
        examples.append({"instruction":json.dumps(prompt,default=str),"output":json.dumps(response)})
        if len(examples)>=limit: break
    return examples


def main():
    p=argparse.ArgumentParser(); p.add_argument("--base-model",default="Qwen/Qwen2.5-0.5B-Instruct"); p.add_argument("--output-dir",default="models/learnbridge-lora"); p.add_argument("--epochs",type=float,default=1); p.add_argument("--learning-rate",type=float,default=2e-4); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--max-training-examples",type=int,default=60); p.add_argument("--seed",type=int,default=42); p.add_argument("--dry-run",action="store_true"); args=p.parse_args()
    examples=build_examples(args.max_training_examples,args.seed)
    if not examples or "learner_profile" not in examples[0]["instruction"]: raise RuntimeError("Instruction data validation failed")
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"training_manifest.json").write_text(json.dumps({"base_model":args.base_model,"examples":len(examples),"trained":not args.dry_run,"seed":args.seed},indent=2))
    (out/"training_sample.jsonl").write_text("\n".join(json.dumps(x) for x in examples[:5]))
    if args.dry_run: return
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc: raise SystemExit("Install requirements-train.txt before training") from exc
    tokenizer=AutoTokenizer.from_pretrained(args.base_model); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.base_model); dataset=Dataset.from_list([{"text":f"Instruction:\n{x['instruction']}\nAnswer:\n{x['output']}"} for x in examples])
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"])
    train_args=TrainingArguments(output_dir=str(out/"checkpoints"),num_train_epochs=args.epochs,learning_rate=args.learning_rate,per_device_train_batch_size=args.batch_size,logging_steps=5,save_strategy="no",report_to=[])
    trainer=SFTTrainer(model=model,args=train_args,train_dataset=dataset,peft_config=config,processing_class=tokenizer); trainer.train(); trainer.model.save_pretrained(out); tokenizer.save_pretrained(out)


if __name__=="__main__": main()
