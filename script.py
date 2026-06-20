from pathlib import Path

# Folder structure
directories = [
    "data/raw",
    "data/processed",

    "notebooks",

    "src/data",
    "src/models",
    "src/retrieval",
    "src/utils",

    "configs",

    "checkpoints",
    "embeddings",

    "serving",

    "demo",

    "tests",

    "results",

    "mlruns",
]

files = [
    "data/README.md",

    "notebooks/01_eda.ipynb",
    "notebooks/02_baseline_mf.ipynb",
    "notebooks/03_ablation.ipynb",

    "src/data/preprocess.py",
    "src/data/dataset.py",
    "src/data/splits.py",

    "src/models/two_tower.py",
    "src/models/baseline_mf.py",
    "src/models/losses.py",

    "src/train.py",
    "src/evaluate.py",

    "src/retrieval/build_index.py",
    "src/retrieval/search.py",

    "src/utils/config.py",
    "src/utils/logging.py",

    "configs/train_config.yaml",
    "configs/retrieval_config.yaml",

    "serving/app.py",
    "serving/schemas.py",
    "serving/model_loader.py",
    "serving/Dockerfile",
    "serving/requirements.txt",

    "demo/streamlit_app.py",

    "tests/test_dataset.py",
    "tests/test_model.py",
    "tests/test_api.py",

    "results/metrics.json",
    "results/latency_benchmark.csv",
    "results/architecture_diagram.png",

    ".gitignore",
    "requirements.txt",
    "docker-compose.yml",
    "README.md",
]

# Create directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Create files
for file in files:
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print("✅ Two-Tower RecSys project structure created successfully!")