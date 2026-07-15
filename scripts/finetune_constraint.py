"""Fine-tune a small LoRA adapter on four responsible constraint labels."""
import argparse
import json
import logging
from pathlib import Path

from learnbridge.constraint_llm import build_balanced_examples
from learnbridge.data_loader import load_resources

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    parser.add_argument(
        "--output-dir", default="models/learnbridge-smollm-constraint-lora"
    )
    parser.add_argument("--examples-per-label", type=int, default=6)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = build_balanced_examples(
        load_resources(), args.examples_per_label, args.seed
    )
    data_path = output_dir / "training_data.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
    manifest = {
        "base_model": args.base_model,
        "examples": len(examples),
        "examples_per_label": args.examples_per_label,
        "epochs": args.epochs,
        "labels": [
            "ACCEPT",
            "REJECT_BUDGET",
            "REJECT_CAPTIONS",
            "REJECT_LEVEL",
        ],
        "status": "dry-run" if args.dry_run else "running",
        "seed": args.seed,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.dry_run:
        LOGGER.info("Validated %s balanced examples", len(examples))
        return

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit("Install requirements-train.txt before training") from error

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    dataset = Dataset.from_list(
        [
            {"prompt": row["prompt"], "completion": row["completion"]}
            for row in examples
        ]
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        max_length=128,
        completion_only_loss=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=8,
        save_total_limit=1,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    result = trainer.train(
        resume_from_checkpoint=True if args.resume_from_checkpoint else None
    )
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    manifest.update(
        {
            "status": "complete",
            "train_loss": float(result.training_loss),
            "global_steps": int(result.global_step),
        }
    )
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOGGER.info("Constraint adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
