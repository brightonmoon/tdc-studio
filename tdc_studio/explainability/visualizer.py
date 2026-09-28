"""RDKit 2D Molecular Heatmap Visualization for Atom Attributions."""

import base64
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def value_to_rgb(val: float) -> Tuple[float, float, float]:
    """Map normalized attribution [-1.0, 1.0] to an intuitive RGB color.

    Positive (> 0): Red hue (hotspot / toxicity liability)
    Neutral (~ 0): Light gray
    Negative (< 0): Green hue (favorable / safety contributor)
    """
    val = float(np.clip(val, -1.0, 1.0))
    if val >= 0:
        # Interpolate from light gray (0.9, 0.9, 0.9) to red (0.95, 0.2, 0.2)
        r = 0.9 + 0.05 * val
        g = 0.9 - 0.70 * val
        b = 0.9 - 0.70 * val
    else:
        # Interpolate from light gray (0.9, 0.9, 0.9) to green (0.2, 0.85, 0.3)
        mag = abs(val)
        r = 0.9 - 0.70 * mag
        g = 0.9 - 0.05 * mag
        b = 0.9 - 0.60 * mag
    return (float(r), float(g), float(b))


class AttributionVisualizer:
    """Renders 2D molecular structures with color-highlighted attribution contours."""

    def __init__(self, width: int = 450, height: int = 350):
        self.width = width
        self.height = height

    def render_svg(
        self,
        smiles: str,
        atom_weights: List[float],
        legend: str = "TDC-Studio XAI Attribution",
    ) -> str:
        """Render high-resolution SVG string highlighting atom attributions."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "<svg></svg>"

        AllChem.Compute2DCoords(mol)

        weights = np.array(atom_weights, dtype=np.float32)
        if len(weights) != mol.GetNumHeavyAtoms():
            # Adjust length if hydrogen atoms or mismatch
            padded = np.zeros(mol.GetNumHeavyAtoms(), dtype=np.float32)
            n_copy = min(len(weights), len(padded))
            padded[:n_copy] = weights[:n_copy]
            weights = padded

        # Normalize weights to [-1, 1]
        max_abs = np.max(np.abs(weights)) if len(weights) > 0 else 1.0
        norm_weights = weights / max_abs if max_abs > 1e-5 else weights

        # Build highlight dictionaries
        atom_cols = {}
        highlight_atoms = []
        atom_radii = {}

        for idx, w in enumerate(norm_weights):
            if abs(w) > 0.08:  # threshold to highlight
                highlight_atoms.append(idx)
                atom_cols[idx] = value_to_rgb(float(w))
                atom_radii[idx] = float(0.35 + 0.25 * min(1.0, abs(w)))

        drawer = rdMolDraw2D.MolDraw2DSVG(self.width, self.height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.legendFontSize = 14

        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_cols,
            highlightAtomRadii=atom_radii,
            legend=legend,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def render_data_uri(
        self,
        smiles: str,
        atom_weights: List[float],
        legend: str = "TDC-Studio XAI Attribution",
    ) -> str:
        """Render SVG as base64 Data URI suitable for direct web / frontend <img> tags."""
        svg_text = self.render_svg(smiles, atom_weights, legend=legend)
        b64_encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64_encoded}"
