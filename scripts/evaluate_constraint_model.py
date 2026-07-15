"""Compare base and LoRA-adapted constraint decisions on held-out resources."""
import argparse
import gc
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from learnbridge.constraint_llm import LABELS, build_balanced_examples, parse_label
from learnbridge.data_loader import load_resources

LOGGER = logging.getLogger(__name__)


def generate(model, tokenizer, examples, device: str, max_new_tokens: int):
    import torch

    model.eval()
    rows = []
    for index, example in enumerate(examples, start=1):
        inputs = tokenizer(
            example["prompt"], return_tensors="pt", truncation=True, max_length=128
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            tokens[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        rows.append(
            {
                "resource_id": example["resource_id"],
                "true_label": example["label"],
                "predicted_label": parse_label(raw),
                "raw_output": raw,
            }
        )
        LOGGER.info("Generated %s/%s", index, len(examples))
    return rows


def metrics(rows):
    truth = [row["true_label"] for row in rows]
    predictions = [row["predicted_label"] or "INVALID" for row in rows]
    valid = [prediction in LABELS for prediction in predictions]
    rejects = [index for index, label in enumerate(truth) if label != "ACCEPT"]
    false_negatives = sum(predictions[index] == "ACCEPT" for index in rejects)
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(
            truth, predictions, labels=LABELS, average="macro", zero_division=0
        ),
        "valid_label_rate": sum(valid) / len(valid),
        "constraint_false_negative_rate": false_negatives / max(1, len(rejects)),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=LABELS + ["INVALID"]
        ).tolist(),
        "confusion_labels": LABELS + ["INVALID"],
    }


def release(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    parser.add_argument(
        "--adapter-dir", default="models/learnbridge-smollm-constraint-lora"
    )
    parser.add_argument("--train-examples-per-label", type=int, default=6)
    parser.add_argument("--eval-examples-per-label", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements-train.txt before evaluation") from error

    resources = load_resources()
    training = build_balanced_examples(
        resources, args.train_examples_per_label, args.seed
    )
    excluded = {row["resource_id"] for row in training}
    evaluation = build_balanced_examples(
        resources,
        args.eval_examples_per_label,
        args.seed + 1000,
        excluded_ids=excluded,
    )
    assert not excluded & {row["resource_id"] for row in evaluation}
    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Completed adapter missing from {adapter_dir}")
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    LOGGER.info("Using %s for %s held-out examples", device, len(evaluation))

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    base_rows = generate(
        base_model, tokenizer, evaluation, device, args.max_new_tokens
    )
    release(base_model)

    adapted_base = AutoModelForCausalLM.from_pretrained(args.base_model)
    adapted_model = PeftModel.from_pretrained(adapted_base, adapter_dir).to(device)
    adapted_rows = generate(
        adapted_model, tokenizer, evaluation, device, args.max_new_tokens
    )
    release(adapted_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(base_rows).to_csv(output_dir / "constraint_base_results.csv", index=False)
    pd.DataFrame(adapted_rows).to_csv(
        output_dir / "constraint_adapted_results.csv", index=False
    )
    summary = {
        "computed": True,
        "held_out_examples": len(evaluation),
        "training_examples": len(training),
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "base": metrics(base_rows),
        "adapted": metrics(adapted_rows),
    }
    (output_dir / "constraint_llm_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    LOGGER.info("Saved comparison to %s", output_dir / "constraint_llm_comparison.json")


if __name__ == "__main__":
    main()
