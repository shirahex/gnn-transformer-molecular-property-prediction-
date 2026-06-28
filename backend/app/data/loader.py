"""DeepChem MoleculeNet dataset loader with fallback to CSV."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from app import config
from app.data.cache import GraphCache
from app.data.preprocessor import smiles_to_graph, smiles_to_tokens

logger = logging.getLogger(__name__)

# Direct download URLs as fallback
FALLBACK_URLS = {
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv",
}


def _get_deepchem_loaders():
    """Lazy import of DeepChem loaders to avoid TensorFlow at import time."""
    from deepchem.molnet import load_bbbp, load_clintox, load_esol, load_freesolv
    return {
        "esol": load_esol,
        "bbbp": load_bbbp,
        "freesolv": load_freesolv,
        "clintox": load_clintox,
    }


class MoleculeDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed molecular graphs."""

    def __init__(self, graphs: list[Data]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


class SMILESDataset(Dataset):
    """Dataset for transformer-based models using SMILES strings."""

    def __init__(self, smiles: list[str], labels: np.ndarray, tokenizer_name: str = config.CHEMBERTA_MODEL_NAME):
        from transformers import AutoTokenizer
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SMILES_LENGTH,
            return_tensors="pt",
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label,
        }


class MoleculeNetDataLoader:
    """Unified loader for MoleculeNet datasets via DeepChem with CSV fallback."""

    def __init__(self, dataset_name: str, model_type: str = "gcn"):
        """Initialize loader.

        Args:
            dataset_name: One of 'esol', 'bbbp', 'freesolv', 'clintox'.
            model_type: One of 'gcn', 'gat', 'chemberta'.
        """
        if dataset_name not in config.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config.DATASETS.keys())}")
        if model_type not in config.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_type}. Available: {config.AVAILABLE_MODELS}")

        self.dataset_name = dataset_name
        self.model_type = model_type
        self.dataset_config = config.DATASETS[dataset_name]
        self.cache = GraphCache()

        # Placeholders
        self.train_graphs: list[Data] = []
        self.val_graphs: list[Data] = []
        self.test_graphs: list[Data] = []
        self.train_smiles: list[str] = []
        self.val_smiles: list[str] = []
        self.test_smiles: list[str] = []
        self.train_labels: np.ndarray = np.array([])
        self.val_labels: np.ndarray = np.array([])
        self.test_labels: np.ndarray = np.array([])

        self.task_type = self.dataset_config["task_type"]
        self.num_classes = None

    def load(self) -> "MoleculeNetDataLoader":
        """Load dataset using DeepChem with CSV fallback."""
        logger.info("Loading %s dataset...", self.dataset_name)

        try:
            self._load_deepchem()
        except Exception as e:
            logger.warning("DeepChem load failed (%s), trying CSV fallback...", e)
            self._load_csv_fallback()

        if self.model_type in ("gcn", "gat"):
            self._preprocess_graphs()
        # For chemberta, we keep SMILES strings

        logger.info(
            "Loaded %s: train=%d, val=%d, test=%d",
            self.dataset_name,
            len(self.train_smiles),
            len(self.val_smiles),
            len(self.test_smiles),
        )
        return self

    def _load_deepchem(self) -> None:
        """Load via DeepChem."""
        loader_map = _get_deepchem_loaders()
        loader_fn = loader_map[self.dataset_name]
        tasks, datasets, transformers = loader_fn(featurizer="Raw", splitter="scaffold")
        train_dataset, val_dataset, test_dataset = datasets

        self.train_smiles = [data.smiles for data in train_dataset.X]
        self.val_smiles = [data.smiles for data in val_dataset.X]
        self.test_smiles = [data.smiles for data in test_dataset.X]

        # Labels might be multi-task; take first task
        self.train_labels = np.array([y[0] if hasattr(y, "__len__") else y for y in train_dataset.y], dtype=np.float32)
        self.val_labels = np.array([y[0] if hasattr(y, "__len__") else y for y in val_dataset.y], dtype=np.float32)
        self.test_labels = np.array([y[0] if hasattr(y, "__len__") else y for y in test_dataset.y], dtype=np.float32)

    def _load_csv_fallback(self) -> None:
        """Load from CSV file."""
        csv_path = config.RAW_DATA_DIR / f"{self.dataset_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found at {csv_path}. "
                f"Please download from {FALLBACK_URLS.get(self.dataset_name, 'unknown')} "
                f"and place it in {config.RAW_DATA_DIR}"
            )

        df = pd.read_csv(csv_path)

        # Column detection
        smiles_col = next(
            (c for c in df.columns if c.lower() in ("smiles", "smiles string", "smile")), None
        )
        if smiles_col is None:
            raise ValueError(f"No SMILES column found in {csv_path}. Columns: {list(df.columns)}")

        # Find target column (usually named after the task or 'measured log solubility in mols per litre')
        target_col = None
        for c in df.columns:
            if c.lower() != smiles_col and c.lower() not in ("compound id", "name", "iupac"):
                target_col = c
                break

        if target_col is None:
            raise ValueError(f"No target column found in {csv_path}")

        smiles = df[smiles_col].astype(str).tolist()
        labels = df[target_col].astype(float).values

        # Simple random split (80/10/10)
        from sklearn.model_selection import train_test_split
        train_s, temp_s, train_l, temp_l = train_test_split(
            smiles, labels, test_size=0.2, random_state=config.RANDOM_SEED
        )
        val_s, test_s, val_l, test_l = train_test_split(
            temp_s, temp_l, test_size=0.5, random_state=config.RANDOM_SEED
        )

        self.train_smiles, self.val_smiles, self.test_smiles = train_s, val_s, test_s
        self.train_labels, self.val_labels, self.test_labels = train_l, val_l, test_l

    def _preprocess_graphs(self) -> None:
        """Convert SMILES to PyG graph objects."""
        self.train_graphs = self._to_graphs(self.train_smiles, self.train_labels)
        self.val_graphs = self._to_graphs(self.val_smiles, self.val_labels)
        self.test_graphs = self._to_graphs(self.test_smiles, self.test_labels)

        # Remove None values (invalid SMILES)
        self.train_graphs = [g for g in self.train_graphs if g is not None]
        self.val_graphs = [g for g in self.val_graphs if g is not None]
        self.test_graphs = [g for g in self.test_graphs if g is not None]

    def _to_graphs(self, smiles_list: list[str], labels: np.ndarray) -> list[Data]:
        """Convert a list of SMILES to graph Data objects."""
        graphs = []
        for smi, lab in zip(smiles_list, labels):
            g = smiles_to_graph(smi, float(lab), self.cache)
            if g is not None:
                graphs.append(g)
        return graphs

    def get_train_loader(self, batch_size: int = config.BATCH_SIZE, shuffle: bool = True):
        """Get training DataLoader."""
        if self.model_type in ("gcn", "gat"):
            return PyGDataLoader(
                MoleculeDataset(self.train_graphs),
                batch_size=batch_size,
                shuffle=shuffle,
            )
        else:  # chemberta
            from torch.utils.data import DataLoader
            return DataLoader(
                SMILESDataset(self.train_smiles, self.train_labels),
                batch_size=batch_size,
                shuffle=shuffle,
            )

    def get_val_loader(self, batch_size: int = config.BATCH_SIZE):
        """Get validation DataLoader."""
        if self.model_type in ("gcn", "gat"):
            return PyGDataLoader(
                MoleculeDataset(self.val_graphs),
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            from torch.utils.data import DataLoader
            return DataLoader(
                SMILESDataset(self.val_smiles, self.val_labels),
                batch_size=batch_size,
                shuffle=False,
            )

    def get_test_loader(self, batch_size: int = config.BATCH_SIZE):
        """Get test DataLoader."""
        if self.model_type in ("gcn", "gat"):
            return PyGDataLoader(
                MoleculeDataset(self.test_graphs),
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            from torch.utils.data import DataLoader
            return DataLoader(
                SMILESDataset(self.test_smiles, self.test_labels),
                batch_size=batch_size,
                shuffle=False,
            )

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "name": self.dataset_config["name"],
            "task_type": self.task_type,
            "train_size": len(self.train_smiles),
            "val_size": len(self.val_smiles),
            "test_size": len(self.test_smiles),
            "num_features": 30 if self.model_type in ("gcn", "gat") else config.MAX_SMILES_LENGTH,
            "description": self.dataset_config["description"],
        }