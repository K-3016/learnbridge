# PR: Build the LearnBridge responsible recommender prototype

## Summary

Implements the complete LearnBridge cold-start educational recommender, evaluation pipeline, Streamlit interface, optional LoRA workflow, tests, documentation, and deployment configuration.

## Approach and trade-offs

- Uses a small fitted TF-IDF model for stable CPU inference.
- Separates popularity, personalized content, and responsible hybrid systems.
- Enforces budget, caption, difficulty, and provider constraints before final selection.
- Uses MMR to expose the relevance-diversity trade-off rather than claiming one universally optimal ranking.
- Keeps optional language-model adaptation downstream of retrieval and disabled by default.
- Uses reproducible synthetic data; no claim is made about real learner outcomes.

## Evidence

- `20 passed` with `pytest -q`.
- `make all` generates data, fits features, validates inference, evaluates all systems, produces pitch assets, and runs tests.
- Computed evaluation results are stored under `data/outputs/`.
- The responsible model reaches 1.000 constraint satisfaction and 1.000 knowledge-gap coverage on the seed-42 synthetic evaluation set.
- Five generated figures visualize model comparison, trade-offs, constraints, coverage, and provider exposure.

## Solo review record

Solo project — self-review performed. Every changed file was inspected with the following checks:

- [x] Tests and expected behavior
- [x] Design and modularity
- [x] Naming, docstrings, comments, and main guards
- [x] Synthetic-data and metric reproducibility
- [x] Budget, captions, difficulty, diversity, and provider behavior
- [x] Metadata-grounded explanations and prohibited claims
- [x] Accessibility, graceful fallback, and deployment startup path
- [x] No secrets, private data, or base-model binaries
- [x] README, pitch, ethics, and AI-use attribution

## Known limitations

- Data and judgments are synthetic.
- Independent approval is unavailable because this is an individual project. An instructor, teaching assistant, or assigned peer should be requested if the grading policy requires a second person’s approval.
- The real LoRA base-versus-adapted comparison remains explicitly uncomputed until actual training and inference outputs exist.
