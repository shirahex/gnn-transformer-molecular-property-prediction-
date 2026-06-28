"""SMILES to molecular graph and token preprocessing utilities."""

import warnings
from typing import Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from transformers import AutoTokenizer

from app import config

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Atom feature dimension
ATOM_FDIM = 30


def _one_hot_encode(value: int, choices: list) -> list[float]:
    """One-hot encode a categorical value."""
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    return encoding


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    """Compute a 30-dimensional feature vector for an atom.

    Features include:
        - Atomic number (one-hot, first 10 elements)
        - Degree (0-5)
        - Formal charge (-1, 0, 1)
        - Chiral tag (4 types)
        - Hybridization (5 types)
        - Aromatic (binary)
        - Is in ring (binary)
        - Mass (scaled)
        - Num explicit Hs (0-3)
    """
    # Atomic number (one-hot for H, C, N, O, F, P, S, Cl, Br, I, other)
    atomic_num = atom.GetAtomicNum()
    atomic_num_feats = _one_hot_encode(atomic_num, [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0])

    # Degree (0-5, 6+)
    degree = atom.GetDegree()
    degree_feats = _one_hot_encode(min(degree, 5), [0, 1, 2, 3, 4, 5])

    # Formal charge bucketed to -1, 0, 1
    charge = atom.GetFormalCharge()
    charge = max(-1, min(1, charge))
    charge_feats = _one_hot_encode(charge, [-1, 0, 1])

    # Chiral tag (4 types: unspecified, tetra CW, tetra CCW, other)
    chiral = atom.GetChiralTag()
    chiral_feats = _one_hot_encode(int(chiral), [0, 1, 2, 3])

    # Hybridization (5 common types)
    hybrid = atom.GetHybridization()
    hybrid_feats = _one_hot_encode(
        int(hybrid),
        [
            int(Chem.HybridizationType.SP),
            int(Chem.HybridizationType.SP2),
            int(Chem.HybridizationType.SP3),
            int(Chem.HybridizationType.SP3D),
            int(Chem.HybridizationType.SP3D2),
        ],
    )

    # Boolean features
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    ring = 1.0 if atom.IsInRing() else 0.0

    # Mass (scaled by 100)
    mass = atom.GetMass() / 100.0

    # Number of explicit hydrogens (0-3)
    num_h = min(atom.GetNumExplicitHs(), 3)
    h_feats = _one_hot_encode(num_h, [0, 1, 2, 3])

    features = (
        atomic_num_feats
        + degree_feats
        + charge_feats
        + chiral_feats
        + hybrid_feats
        + [aromatic, ring, mass]
        + h_feats
    )

    # Pad or truncate to exactly ATOM_FDIM
    features = features[:ATOM_FDIM]
    features += [0.0] * (ATOM_FDIM - len(features))

    return np.array(features, dtype=np.float32)


def _bond_features(bond: Chem.Bond) -> np.ndarray:
    """Compute a 6-dimensional feature vector for a bond."""
    bt = bond.GetBondType()
    bond_type_feats = _one_hot_encode(
        int(bt),
        [
            int(Chem.BondType.SINGLE),
            int(Chem.BondType.DOUBLE),
            int(Chem.BondType.TRIPLE),
            int(Chem.BondType.AROMATIC),
        ],
    )
    conjugated = 1.0 if bond.GetIsConjugated() else 0.0
    ring = 1.0 if bond.IsInRing() else 0.0
    stereo = _one_hot_encode(int(bond.GetStereo()), [0, 1, 2, 3, 4, 5])

    features = bond_type_feats + [conjugated, ring] + stereo
    return np.array(features[:6], dtype=np.float32)


def smiles_to_graph(
    smiles: str, label: Optional[float | int] = None, cache=None
) -> Optional[Data]:
    """Convert a SMILES string to a PyG Data object.

    Args:
        smiles: SMILES string.
        label: Optional target label.
        cache: Optional GraphCache instance.

    Returns:
        PyG Data object with x (node features), edge_index, edge_attr, y,
        and smiles, or None if parsing fails.
    """
    if cache is not None:
        cached = cache.get(smiles)
        if cached is not None:
            if label is not None:
                cached.y = torch.tensor([label], dtype=torch.float)
            return cached

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.stack(atom_feats), dtype=torch.float)

    # Edges
    edge_list = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append((i, j))
        edge_list.append((j, i))  # Undirected
        feat = _bond_features(bond)
        edge_feats.append(feat)
        edge_feats.append(feat)

    if len(edge_list) == 0:
        # Isolated atom — add self-loop
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_feats), dtype=torch.float)

    # Label
    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
    else:
        y = torch.tensor([0.0], dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        smiles=smiles,
        num_nodes=x.size(0),
    )

    if cache is not None:
        cache.put(smiles, data)

    return data


def smiles_to_tokens(smiles_list: list[str]) -> dict[str, torch.Tensor]:
    """Tokenize SMILES strings for ChemBERTa.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Dictionary with 'input_ids' tensor.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.CHEMBERTA_MODEL_NAME)
    encoded = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=config.MAX_SMILES_LENGTH,
        return_tensors="pt",
    )
    return encoded
