"""Base DataModule for Therapeutics Data Commons datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from tdc_studio.core.exceptions import DataPipelineError


class MolecularDataset(Dataset):
    """Generic in-memory PyTorch Dataset for molecular samples."""

    def __init__(self, samples: list[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class BaseTDCDataModule(ABC):
    """Abstract base class for TDC dataset loaders and splitters."""

    def __init__(
        self,
        dataset_name: str,
        split_type: str = "scaffold",
        seed: int = 42,
        task_type: str = "regression",
        metric_name: str = "mae",
        synthetic_df: Optional[pd.DataFrame] = None,
    ):
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.seed = seed
        self.task_type = task_type
        self.metric_name = metric_name
        self.synthetic_df = synthetic_df

        self.splits: Dict[str, pd.DataFrame] = {}
        self.is_prepared = False

    @abstractmethod
    def prepare_data(self) -> None:
        """Download or prepare the dataset splits."""
        pass

    @abstractmethod
    def setup_loaders(
        self, batch_size: int = 32, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_loader, val_loader, test_loader)."""
        pass

    def check_prepared(self) -> None:
        if not self.is_prepared:
            raise DataPipelineError(
                f"DataModule for '{self.dataset_name}' must call prepare_data() before setup_loaders()."
            )
