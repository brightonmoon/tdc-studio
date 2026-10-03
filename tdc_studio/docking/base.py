"""Abstract Base Interfaces and Data Models for Multi-Provider 3D Docking."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DockingPose:
    """Individual 3D binding conformation and associated score."""
    pose_id: int
    affinity_kcal_mol: float
    rmsd_lb: float = 0.0
    rmsd_ub: float = 0.0
    pdbqt_or_pdb_block: Optional[str] = None


@dataclass
class DockingResult:
    """Comprehensive result of a 3D molecular docking execution."""
    engine: str
    success: bool
    top_affinity: float                 # Best binding energy (kcal/mol; lower/more negative is stronger)
    poses: List[DockingPose]
    ligand_smiles: str
    receptor_path: str
    execution_time_sec: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "success": self.success,
            "top_affinity": self.top_affinity,
            "num_poses": len(self.poses),
            "poses": [asdict(p) for p in self.poses],
            "ligand_smiles": self.ligand_smiles,
            "receptor_path": str(self.receptor_path),
            "execution_time_sec": round(self.execution_time_sec, 3),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class BaseDockingEngine(ABC):
    """Abstract base provider for 3D molecular docking engines."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine runtime binaries, python packages, or cloud credentials are ready."""
        pass

    @abstractmethod
    def dock(
        self,
        ligand_smiles: str,
        receptor_path: Union[str, Path],
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        num_poses: int = 9,
        exhaustiveness: int = 8,
        **kwargs: Any,
    ) -> DockingResult:
        """Execute 3D docking of a ligand against a target protein structure.

        Args:
            ligand_smiles: SMILES string of the drug candidate.
            receptor_path: Path to target structure (PDB or PDBQT).
            center: (x, y, z) 3D coordinate of the binding pocket center in Angstroms.
            box_size: (size_x, size_y, size_z) search grid dimensions in Angstroms.
            num_poses: Number of output binding modes to generate.
            exhaustiveness: Search depth / Monte Carlo exhaustiveness.
            **kwargs: Provider-specific execution options.

        Returns:
            DockingResult containing top binding energy and generated 3D poses.
        """
        pass
