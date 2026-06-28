"""
src/utils/config.py
--------------------
Typed config objects loaded from configs/train_config.yaml.
Keeps train.py free of magic numbers and makes every run reproducible/loggable.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    id_maps_file: str = "id_maps.json"
    num_eval_negatives: int = 99   # NCF-style sampled evaluation (1 positive + N negatives)


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 512
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1       # for in-batch contrastive loss
    eval_every: int = 1
    top_k: int = 10
    seed: int = 42
    device: str = "cpu"            # "cuda" if available
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    log_dir: str = "results/logs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(instance, overrides: dict):
    """Overwrite dataclass fields in-place from a plain dict (one level deep)."""
    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            raise ValueError(f"Unknown config key '{key}' for {type(instance).__name__}")
    return instance


def load_config(path: str) -> Config:
    """
    Load a YAML file like configs/train_config.yaml into a typed Config object.
    Missing sections/keys fall back to the dataclass defaults above.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    if "data" in raw:
        _merge_dataclass(cfg.data, raw["data"])
    if "model" in raw:
        _merge_dataclass(cfg.model, raw["model"])
    if "train" in raw:
        _merge_dataclass(cfg.train, raw["train"])

    return cfg


def config_to_dict(cfg: Config) -> dict:
    """Useful for dumping the exact run config into results/metrics.json."""
    return dataclasses.asdict(cfg)


if __name__ == "__main__":
    # quick manual check: python config.py path/to/train_config.yaml
    import sys
    cfg = load_config(sys.argv[1]) if len(sys.argv) > 1 else Config()
    print(cfg)