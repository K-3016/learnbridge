FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-deps -e .
COPY . .
RUN PYTHONPATH=src python scripts/make_dataset.py && PYTHONPATH=src python scripts/build_features.py
EXPOSE 7860
CMD ["streamlit","run","main.py","--server.address=0.0.0.0","--server.port=7860"]
