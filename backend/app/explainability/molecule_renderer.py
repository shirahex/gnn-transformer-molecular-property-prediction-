"""RDKit-based 2D molecular structure rendering with attention highlighting."""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


class MoleculeRenderer:
    """Render 2D molecular structures with optional atom highlighting."""

    def __init__(self, width: int = 400, height: int = 400) -> None:
        self.width = width
        self.height = height

    def render(
        self,
        smiles: str,
        atom_scores: Optional[dict[int, float]] = None,
        highlight_functional_groups: bool = True,
    ) -> Optional[bytes]:
        """Render a molecule as PNG bytes with optional attention highlighting.

        Args:
            smiles: SMILES string.
            atom_scores: Dictionary mapping atom index -> score (0-1) for coloring.
            highlight_functional_groups: Whether to detect and label functional groups.

        Returns:
            PNG image bytes or None if rendering fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error("Failed to parse SMILES: %s", smiles)
            return None

        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        # Generate highlight colors based on attention scores
        highlight_atoms = {}
        highlight_bonds = {}

        if atom_scores:
            for atom_idx, score in atom_scores.items():
                # Map score (0-1) to color (white -> red -> dark red)
                r = min(255, int(255 * score))
                g = int(255 * (1 - score))
                b = int(255 * (1 - score) * 0.5)
                color = (r / 255, g / 255, b / 255, 0.6)
                highlight_atoms[int(atom_idx)] = color

        # Detect functional groups for additional highlighting
        if highlight_functional_groups:
            fg_highlights = self._detect_functional_groups(mol)
            for atom_idx, fg_name in fg_highlights.items():
                if atom_idx not in highlight_atoms:
                    # Use blue tint for functional groups
                    highlight_atoms[atom_idx] = (0.3, 0.5, 1.0, 0.4)

        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(self.width, self.height)
            draw_options = drawer.drawOptions()
            draw_options.addAtomIndices = False
            draw_options.annotationFontScale = 0.7

            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_atoms.keys()) if highlight_atoms else [],
                highlightAtomColors=highlight_atoms if highlight_atoms else None,
                highlightBonds=list(highlight_bonds.keys()) if highlight_bonds else [],
                highlightBondColors=highlight_bonds if highlight_bonds else None,
            )
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
            return png_bytes
        except Exception as e:
            logger.error("Cairo drawer failed (%s), falling back to PIL", e)
            return self._render_fallback(mol, highlight_atoms)

    def _render_fallback(
        self, mol: Chem.Mol, highlight_atoms: dict[int, tuple]
    ) -> Optional[bytes]:
        """Fallback rendering using PIL."""
        try:
            img = Draw.MolToImage(mol, size=(self.width, self.height))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.error("Fallback rendering also failed: %s", e)
            return None

    def _detect_functional_groups(self, mol: Chem.Mol) -> dict[int, str]:
        """Detect common functional groups and return atom mappings.

        Returns:
            Dictionary mapping atom index to functional group name.
        """
        result = {}

        # SMARTS patterns for common functional groups
        patterns = {
            "hydroxyl": "[OX2H]",
            "carbonyl": "[#6X3]=[OX1]",
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "amide": "[NX3][CX3](=[OX1])",
            "carboxylic_acid": "[CX3](=O)[OX2H1]",
            "nitro": "[NX3](=[OX1])[OX1]",
            "sulfonyl": "[SX4](=[OX1])(=[OX1])",
            "ether": "[OD2]([#6])[#6]",
            "alkene": "[CX3]=[CX3]",
            "alkyne": "[CX2]#[CX2]",
            "aromatic": "[a]",
        }

        for fg_name, smarts in patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    for atom_idx in match:
                        result[atom_idx] = fg_name
            except Exception:
                continue

        return result

    def render_comparison(
        self,
        smiles: str,
        gcn_scores: Optional[dict[int, float]] = None,
        gat_scores: Optional[dict[int, float]] = None,
        chemberta_scores: Optional[dict[int, float]] = None,
    ) -> Optional[bytes]:
        """Render side-by-side comparison of attention from 3 models.

        Returns:
            PNG image bytes.
        """
        images = []
        labels = []

        for label, scores in [
            ("GCN", gcn_scores),
            ("GAT", gat_scores),
            ("ChemBERTa", chemberta_scores),
        ]:
            img_bytes = self.render(smiles, atom_scores=scores, highlight_functional_groups=False)
            if img_bytes:
                images.append(Image.open(io.BytesIO(img_bytes)))
                labels.append(label)

        if not images:
            return None

        # Create combined image
        total_width = sum(img.width for img in images) + 20 * (len(images) - 1)
        max_height = max(img.height for img in images) + 40  # Extra for labels

        combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for img, label in zip(images, labels):
            combined.paste(img, (x_offset, 40))
            x_offset += img.width + 20

        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        return buf.getvalue()
