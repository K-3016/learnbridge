# Model artifacts

`tfidf_features.joblib` is the fitted TF-IDF model used for deployed application inference. It is generated reproducibly by `make features` and committed because it is small.

`learnbridge-smollm-constraint-lora/` contains the final PEFT LoRA adapter and tokenizer configuration produced by `scripts/finetune_constraint.py` from `HuggingFaceTB/SmolLM2-135M-Instruct`. The directory intentionally contains no base-model weights. Its `training_manifest.json` records the completed 24-example, two-epoch run, and the held-out comparison is saved at `data/outputs/constraint_llm_comparison.json`.

The adapter is retained as hackathon experiment evidence. It is not loaded by the deployed Streamlit app and does not enforce hard constraints; deterministic recommendation logic remains authoritative.
