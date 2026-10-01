"""DualModalDrugEncoder — 2D Topology & 3D Conformation Fusion Encoder for DTA.

Design rationale:
- 1D SMILES language models (e.g. ChemBERTa) tend to overfit on specific scaffold BPE substrings
  when fine-tuned on Cold-Drug splits.
- Pure 2D graph GNNs ignore 3D stereochemistry, spatial shape, and binding conformation.
- DualModalDrugEncoder combines:
    1. 2D Topological invariant representation (GNN / Morgan topological fingerprint)
    2. 3D Spatial conformation representation (RDKit ETKDGv3 3D coordinates & RBF distance embedding)
  fused via an adaptive cross-modal gating mechanism:
    gate = sigmoid(Linear([h_2D; h_3D]))
    h_out = gate * h_2D + (1 - gate) * h_3D

Registered as "dual_modal_drug_encoder" and "dual_modal_drug" in MODELS.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS

logger = logging.getLogger("tdc_studio.models.dti")


def generate_3d_coordinates(smiles: str, num_attempts: int = 3) -> Optional[torch.Tensor]:
    """Generate 3D Cartesian coordinates for a SMILES string using RDKit ETKDGv3.

    Args:
        smiles: Valid SMILES string.
        num_attempts: Conformation embedding attempts before giving up.

    Returns:
        FloatTensor [N_atoms, 3] of 3D coordinates, or None if failed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42

        res = AllChem.EmbedMolecule(mol, params)
        if res < 0:
            # Fallback to random coordinates followed by UFF optimization
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res < 0:
                return None

        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass

        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        coords = []
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        return torch.tensor(coords, dtype=torch.float32)
    except Exception as exc:
        logger.debug("Failed 3D coordinate generation for %s: %s", smiles, exc)
        return None


class SpatialRBFDistanceEncoder(nn.Module):
    """Encodes 3D interatomic pairwise distances into Radial Basis Function (RBF) kernels."""

    def __init__(self, num_rbf: int = 32, d_min: float = 0.5, d_max: float = 15.0, hidden_dim: int = 256):
        super().__init__()
        self.num_rbf = num_rbf
        centers = torch.linspace(d_min, d_max, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / ((d_max - d_min) / num_rbf) ** 2

        self.proj = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute RBF distance representation from 3D coordinates.

        Args:
            coords: FloatTensor [N, 3]

        Returns:
            FloatTensor [hidden_dim] (pooled spatial distance features)
        """
        if coords.dim() == 2:
            # Pairwise Euclidean distance matrix [N, N]
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 3]
            dist = torch.norm(diff, dim=-1)  # [N, N]
        else:
            dist = coords

        # RBF expansion [N, N, num_rbf]
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)
        pooled_rbf = rbf.mean(dim=(0, 1))  # [num_rbf]
        return self.proj(pooled_rbf)


@MODELS.register("dual_modal_drug_encoder")
@MODELS.register("dual_modal_drug")
class DualModalDrugEncoder(nn.Module):
    """Dual-modal molecular encoder unifying 2D topology and 3D spatial conformation.

    Args:
        config: Dict with keys:
            hidden_dim          : Internal hidden representation dimension (default 256).
            out_dim             : Final projected output dimension (default 256).
            num_rbf             : Number of 3D distance RBF centers (default 32).
            use_3d_coordinates  : Whether to generate and fuse 3D conformation (default True).
            dropout             : Dropout probability (default 0.1).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}
        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.out_dim = cfg.get("out_dim", 256)
        self.use_3d = cfg.get("use_3d_coordinates", True)
        dropout = cfg.get("dropout", 0.1)

        # 1. 2D Topological Fingerprint / Graph projection
        self.topo_proj = nn.Sequential(
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 2. 3D Conformation branch
        self.spatial_encoder = SpatialRBFDistanceEncoder(
            num_rbf=cfg.get("num_rbf", 32),
            hidden_dim=self.hidden_dim,
        )

        # 3. Cross-modal Adaptive Gating Fusion
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        self.post_norm = nn.LayerNorm(self.hidden_dim)

        # Final projection to out_dim
        if self.out_dim != self.hidden_dim:
            self.final_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.final_proj = nn.Identity()

        # Cache for 3D coordinates & Morgan fingerprints
        self._coord_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._fp_cache: Dict[str, torch.Tensor] = {}

    def _compute_morgan_fp(self, smiles: str) -> torch.Tensor:
        """Compute 1024-bit Morgan Fingerprint as invariant 2D topological representation."""
        if smiles in self._fp_cache:
            return self._fp_cache[smiles]

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                arr = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                fp_tensor = torch.tensor(list(arr), dtype=torch.float32)
            else:
                fp_tensor = torch.zeros(1024, dtype=torch.float32)
        except Exception:
            fp_tensor = torch.zeros(1024, dtype=torch.float32)

        self._fp_cache[smiles] = fp_tensor
        return fp_tensor

    def _get_3d_coordinates(self, smiles: str) -> Optional[torch.Tensor]:
        """Fetch or generate 3D Cartesian coordinates with caching."""
        if smiles in self._coord_cache:
            return self._coord_cache[smiles]
        coords = generate_3d_coordinates(smiles)
        self._coord_cache[smiles] = coords
        return coords

    def encode_smiles_list(self, smiles_list: List[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of SMILES into dual-modal 2D+3D representations.

        Args:
            smiles_list: List of SMILES strings.
            device: Target torch device.

        Returns:
            FloatTensor [B, out_dim]
        """
        topo_tensors = [self._compute_morgan_fp(s) for s in smiles_list]
        h_topo_batch = torch.stack(topo_tensors, dim=0).to(device)  # [B, 1024]
        h_2d = self.topo_proj(h_topo_batch)  # [B, hidden_dim]

        if not self.use_3d:
            return self.final_proj(self.post_norm(h_2d))

        # Compute 3D conformation representations
        spatial_list = []
        for s in smiles_list:
            coords = self._get_3d_coordinates(s)
            if coords is not None and coords.size(0) > 1:
                s_feat = self.spatial_encoder(coords.to(device))
            else:
                # Fallback to zero 3D perturbation if 3D conformer fails
                s_feat = torch.zeros(self.hidden_dim, device=device)
            spatial_list.append(s_feat)

        h_3d = torch.stack(spatial_list, dim=0)  # [B, hidden_dim]

        # Adaptive Gating
        concat_feat = torch.cat([h_2d, h_3d], dim=-1)  # [B, hidden_dim * 2]
        gate = self.gate_layer(concat_feat)  # [B, hidden_dim]

        h_fused = gate * h_2d + (1.0 - gate) * h_3d
        h_norm = self.post_norm(h_fused)
        return self.final_proj(h_norm)

    def extract_features(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Extract drug features from batch dict matching GraphDTAModel interface."""
        device = next(self.parameters()).device

        if "drug_smiles_str" in batch:
            smiles_list = batch["drug_smiles_str"]
        elif "smiles" in batch:
            smiles_list = batch["smiles"]
        elif "drug_graph" in batch and hasattr(batch["drug_graph"], "smiles"):
            smiles_list = batch["drug_graph"].smiles
        else:
            raise KeyError("Batch must contain 'drug_smiles_str' or 'smiles' for DualModalDrugEncoder.")

        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        h = self.encode_smiles_list(smiles_list, device=device)
        if return_sequence:
            return h.unsqueeze(1)
        return h

    def forward(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Standard forward pass."""
        return self.extract_features(batch, return_sequence=return_sequence)
