"""Model Explainability: Integrated Gradients Atom and Bond Attribution for Molecular Graphs.

References:
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.
- Sanchez-Lengeling, B. et al. (2020). Evaluating attribution for graph neural networks. NeurIPS.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from tdc_studio.data.transforms import SmilesToGraphTransform


class MolecularExplainer:
    """Computes atom-level and substructure-level attribution scores using Integrated Gradients."""

    def __init__(
        self,
        model: torch.nn.Module,
        steps: int = 30,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.steps = steps
        self.device = device
        self.graph_transform = SmilesToGraphTransform()

    def attribute(
        self,
        smiles: str,
        target_idx: int = 0,
        baseline_type: str = "zero",
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute atom-level attribution weights for a given molecule SMILES.

        Args:
            smiles: Input SMILES string.
            target_idx: Output index to attribute (for multi-task heads).
            baseline_type: 'zero' (zero atom features) or 'carbon' (carbon-only features).
            steps: Optional override for number of interpolation steps.

        Returns:
            Dictionary containing:
                - 'smiles': canonical SMILES
                - 'num_atoms': number of heavy atoms
                - 'atom_attributions': 1D array of shape [num_atoms] (positive = liability/increase)
                - 'normalized_attributions': normalized to [-1.0, 1.0]
                - 'predicted_score': scalar prediction
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        canon_s = Chem.MolToSmiles(mol)
        graph = self.graph_transform(canon_s)
        if graph is None:
            raise ValueError(f"Could not convert {smiles} to molecular graph.")

        num_atoms = graph.x.size(0)
        x_target = graph.x.clone().detach().to(self.device).float()
        x_target.requires_grad = True

        # 1. Establish baseline
        if baseline_type == "zero":
            x_baseline = torch.zeros_like(x_target)
        else:
            x_baseline = torch.zeros_like(x_target)
            # Set carbon atomic number feature if first dim is atomic num
            x_baseline[:, 0] = 6.0

        # 2. Compute path integral gradients
        accumulated_grads = torch.zeros_like(x_target)

        # Scale interpolation steps
        n_steps = steps if steps is not None else self.steps
        alphas = torch.linspace(0.0, 1.0, n_steps, device=self.device)

        edge_index = graph.edge_index.to(self.device)
        edge_attr = graph.edge_attr.to(self.device) if graph.edge_attr is not None else None

        # Predict original target score
        with torch.no_grad():
            single_batch = Data(x=x_target, edge_index=edge_index, edge_attr=edge_attr)
            batch_input = {"drug_graph": Batch.from_data_list([single_batch])}
            pred_raw = self.model(batch_input)
            if pred_raw.ndim > 1:
                pred_val = float(pred_raw[0, target_idx].item())
            else:
                pred_val = float(pred_raw.item())

        for alpha in alphas:
            x_interp = x_baseline + alpha * (x_target - x_baseline)
            x_interp = x_interp.clone().detach().requires_grad_(True)

            interp_graph = Data(x=x_interp, edge_index=edge_index, edge_attr=edge_attr)
            batch = {"drug_graph": Batch.from_data_list([interp_graph])}

            output = self.model(batch)
            if output.ndim > 1:
                target_score = output[0, target_idx]
            else:
                target_score = output[0] if output.ndim == 1 else output

            # Backward pass to get gradients w.r.t interpolated node features
            self.model.zero_grad()
            if x_interp.grad is not None:
                x_interp.grad.zero_()
            target_score.backward()

            if x_interp.grad is not None:
                accumulated_grads += x_interp.grad

        avg_grads = accumulated_grads / float(self.steps)
        integrated_grads = (x_target - x_baseline) * avg_grads

        # Sum over 14 atomic feature channels to get per-atom attribution
        atom_attributions = integrated_grads.sum(dim=-1).detach().cpu().numpy()

        # Normalized attributions to range [-1.0, 1.0] for clear visualization
        max_abs = np.max(np.abs(atom_attributions)) if len(atom_attributions) > 0 else 1.0
        if max_abs < 1e-6:
            normalized_attr = np.zeros_like(atom_attributions)
        else:
            normalized_attr = atom_attributions / max_abs

        # Identify top toxicophore / hotspot atom indices (positive attribution)
        hotspot_atoms = [int(i) for i in np.where(normalized_attr > 0.25)[0]]

        return {
            "smiles": canon_s,
            "num_atoms": num_atoms,
            "atom_attributions": atom_attributions.tolist(),
            "normalized_attributions": normalized_attr.tolist(),
            "predicted_score": pred_val,
            "hotspot_atoms": hotspot_atoms,
        }
