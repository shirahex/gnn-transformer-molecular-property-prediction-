"""Helper utilities for validation and formatting."""

import logging
import re
from typing import Optional

from rdkit import Chem

logger = logging.getLogger(__name__)


def validate_smiles(smiles: str) -> tuple[bool, Optional[str]]:
    """Validate a SMILES string.

    Args:
        smiles: SMILES string to validate.

    Returns:
        Tuple of (is_valid, error_message_or_None).
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES string is empty or invalid"

    smiles = smiles.strip()
    if len(smiles) == 0:
        return False, "SMILES string is empty"

    if len(smiles) > 1000:
        return False, "SMILES string is too long (> 1000 characters)"

    # Check for invalid characters
    allowed = set("BCNOSPFIabcdefgHhijklmnoPpQqRrSsTtUuVvWwXxYyZz0123456789=@.#$%&!+-[](){}//\\\\~,;:")
    invalid_chars = set(smiles) - allowed
    if invalid_chars:
        return False, f"Invalid characters in SMILES: {''.join(invalid_chars)}"

    # Try to parse with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Could not parse SMILES string"

    return True, None


def get_molecule_info(smiles: str) -> dict:
    """Get basic info about a molecule from its SMILES.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary with molecular weight, formula, num_atoms, num_bonds.
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    mol = Chem.AddHs(mol)
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
    }
