"""Disk caching for preprocessed molecular graphs."""

import hashlib
import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data

from app import config


class GraphCache:
    """Simple disk cache for preprocessed PyG Data objects.

    Caches graphs by SMILES hash to avoid re-processing between runs.
    """

    def __init__(self, cache_dir: Path = config.CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, Data] = {}

    def _hash(self, smiles: str) -> str:
        """Create a deterministic hash for a SMILES string."""
        return hashlib.md5(smiles.encode("utf-8")).hexdigest()

    def _path(self, smiles: str) -> Path:
        """Get cache file path for a SMILES string."""
        return self.cache_dir / f"{self._hash(smiles)}.pkl"

    def get(self, smiles: str) -> Data | None:
        """Retrieve cached graph if it exists.

        Args:
            smiles: SMILES string.

        Returns:
            Cached Data object or None.
        """
        # Check memory cache first
        if smiles in self._memory_cache:
            return self._memory_cache[smiles]

        path = self._path(smiles)
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._memory_cache[smiles] = data
            return data
        return None

    def put(self, smiles: str, data: Data) -> None:
        """Cache a graph object.

        Args:
            smiles: SMILES string.
            data: PyG Data object.
        """
        self._memory_cache[smiles] = data
        path = self._path(smiles)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def clear(self) -> None:
        """Clear all cached graphs."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
