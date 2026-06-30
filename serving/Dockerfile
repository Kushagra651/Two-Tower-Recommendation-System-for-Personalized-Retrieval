# Two-Tower Recommender — inference API container
FROM python:3.11-slim

WORKDIR /app

# Install only what's needed to serve (see serving/requirements.txt)
COPY serving/requirements.txt ./serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

# Code needed at inference time
COPY src/models ./src/models
COPY src/utils ./src/utils
COPY src/__init__.py ./src/__init__.py
COPY serving ./serving

# Artifacts produced by training/build_index — mount or COPY at build/run time
COPY configs/train_config.yaml ./configs/train_config.yaml
COPY checkpoints ./checkpoints
COPY embeddings ./embeddings
COPY data/processed/id_maps.json ./data/processed/id_maps.json

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]