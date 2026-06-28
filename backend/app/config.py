"""Configuration settings for the Molecular Property Prediction backend."""

import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MOLECULAR_DATA_DIR", str(BASE_DIR.parent / "data")))
MODEL_DIR = Path(os.getenv("MOLECULAR_MODEL_DIR", str(BASE_DIR.parent / "models")))
OUTPUT_DIR = Path(os.getenv("MOLECULAR_OUTPUT_DIR", str(BASE_DIR.parent / "outputs")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = os.getenv("MOLECULAR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = int(os.getenv("MOLECULAR_BATCH_SIZE", "32"))
EPOCHS: int = int(os.getenv("MOLECULAR_EPOCHS", "50"))
LEARNING_RATE: float = float(os.getenv("MOLECULAR_LR", "0.001"))
WEIGHT_DECAY: float = float(os.getenv("MOLECULAR_WD", "0.0"))
DROPOUT: float = float(os.getenv("MOLECULAR_DROPOUT", "0.2"))
HIDDEN_DIM: int = int(os.getenv("MOLECULAR_HIDDEN_DIM", "128"))
NUM_LAYERS: int = int(os.getenv("MOLECULAR_NUM_LAYERS", "3"))
NUM_HEADS: int = int(os.getenv("MOLECULAR_NUM_HEADS", "4"))

# Per-model learning rate defaults. Pretrained transformers need a much
# smaller fine-tuning LR than randomly-initialized GNNs trained from
# scratch — 0.001 is reasonable for GCN/GAT but will wreck ChemBERTa's
# pretrained weights almost immediately.
MODEL_LEARNING_RATES: dict[str, float] = {
    "gcn": 0.001,
    "gat": 0.001,
    "chemberta": 2e-5,
}

# ---------------------------------------------------------------------------
# Model-specific
# ---------------------------------------------------------------------------
CHEMBERTA_MODEL_NAME: str = "seyonec/ChemBERTa-zinc-base-v1"
MAX_SMILES_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE: int = 10
LR_SCHEDULER_FACTOR: float = 0.5
LR_SCHEDULER_PATIENCE: int = 5
MIN_LR: float = 1e-6

# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
MC_DROPOUT_PASSES: int = 50
CONFIDENCE_LEVEL: float = 0.95

# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------
WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "molecular-property-prediction")
USE_WANDB: bool = bool(WANDB_API_KEY)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("MOLECULAR_API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("MOLECULAR_API_PORT", "8000"))
CORS_ORIGINS: list[str] = os.getenv("MOLECULAR_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "esol": {
        "name": "ESOL",
        "description": "Delaney solubility dataset — aqueous solubility of molecules",
        "task_type": "regression",
        "metric": "rmse",
        "source": "deepchem",
        "loader": "load_esol",
    },
    "bbbp": {
        "name": "BBBP",
        "description": "Blood-Brain Barrier Penetration — binary classification",
        "task_type": "classification",
        "metric": "roc_auc",
        "source": "deepchem",
        "loader": "load_bbbp",
    },
    "freesolv": {
        "name": "FreeSolv",
        "description": "Hydration free energy of molecules in water",
        "task_type": "regression",
        "metric": "rmse",
        "source": "deepchem",
        "loader": "load_freesolv",
    },
    "clintox": {
        "name": "ClinTox",
        "description": "Clinical toxicity — FDA-approved vs toxic drugs",
        "task_type": "classification",
        "metric": "roc_auc",
        "source": "deepchem",
        "loader": "load_clintox",
    },
}

AVAILABLE_MODELS: list[str] = ["gcn", "gat", "chemberta"]