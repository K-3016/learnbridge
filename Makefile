.PHONY: install data features train evaluate test app clean all
PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src
install:
	$(PYTHON) -m pip install -r requirements-dev.txt
data:
	$(PYTHON) scripts/make_dataset.py
features: data
	$(PYTHON) scripts/build_features.py
train: features
	$(PYTHON) scripts/model.py
evaluate: train
	$(PYTHON) scripts/evaluate.py
	$(PYTHON) scripts/generate_pitch_assets.py
	$(PYTHON) scripts/evaluate_llm.py
test:
	$(PYTHON) -m pytest -q
app:
	streamlit run main.py
clean:
	rm -f data/processed/*.csv data/processed/*.json models/*.joblib data/outputs/*.csv data/outputs/*.json data/outputs/figures/*.png
all: data features train evaluate test
