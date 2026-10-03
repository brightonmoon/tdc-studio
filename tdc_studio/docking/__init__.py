"""Multi-Provider 3D Molecular Docking Bridge and Structure Downloader."""

from typing import Any

from tdc_studio.docking.base import BaseDockingEngine, DockingPose, DockingResult
from tdc_studio.docking.cloud_clients import (
    BioNeMoDiffDockEngine,
    NeurosnapEngine,
    TamarindEngine,
)
from tdc_studio.docking.local_vina import AutoDockVinaEngine
from tdc_studio.docking.structure_fetcher import StructureFetcher


def get_docking_engine(provider: str = "vina", **kwargs: Any) -> BaseDockingEngine:
    """Factory creating docking engine instance for the requested provider.

    Supported providers:
        - "vina", "autodock_vina"  : Local AutoDock Vina engine
        - "smina"                  : Local Smina engine (Vina fork with custom scoring)
        - "bionemo", "diffdock"    : NVIDIA BioNeMo NIM DiffDock Cloud API
        - "neurosnap"              : Neurosnap Cloud DiffDock API
        - "tamarind"               : Tamarind Bio Cloud DiffDock API

    Args:
        provider: Name of docking provider.
        **kwargs: Provider-specific configuration arguments.

    Returns:
        Instance of BaseDockingEngine subclass.
    """
    prov = provider.lower().strip()
    if prov in ("vina", "autodock_vina"):
        return AutoDockVinaEngine(prefer_smina=False, **kwargs)
    elif prov == "smina":
        return AutoDockVinaEngine(prefer_smina=True, **kwargs)
    elif prov in ("bionemo", "diffdock", "nvidia"):
        return BioNeMoDiffDockEngine(**kwargs)
    elif prov == "neurosnap":
        return NeurosnapEngine(**kwargs)
    elif prov == "tamarind":
        return TamarindEngine(**kwargs)
    else:
        raise ValueError(
            f"Unsupported docking provider '{provider}'. "
            f"Supported options: 'vina', 'smina', 'bionemo', 'neurosnap', 'tamarind'."
        )


__all__ = [
    "BaseDockingEngine",
    "DockingPose",
    "DockingResult",
    "AutoDockVinaEngine",
    "BioNeMoDiffDockEngine",
    "NeurosnapEngine",
    "TamarindEngine",
    "StructureFetcher",
    "get_docking_engine",
]
