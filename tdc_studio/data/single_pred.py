"""Single-prediction dataset loader (ADMET, Tox, HTS)."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule, MolecularDataset
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import SmilesToGraphTransform


@DATASETS.register("admet_loader")
class ADMETDataModule(BaseTDCDataModule):
    """ADMET dataset module converting SMILES into molecular graphs."""

    def __init__(
        self,
        dataset_name: str = "caco2_wang",
        split_type: str = "scaffold",
        seed: int = 42,
        task_type: str = "regression",
        metric_name: str = "mae",
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type=task_type,
            metric_name=metric_name,
            synthetic_df=synthetic_df,
        )
        self.graph_transform = SmilesToGraphTransform()

    def prepare_data(self) -> None:
        """Load dataset from synthetic data or TDC library."""
        if self.synthetic_df is not None:
            df = self.synthetic_df
            n = len(df)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            self.splits = {
                "train": df.iloc[:n_train],
                "valid": df.iloc[n_train : n_train + n_val],
                "test": df.iloc[n_train + n_val :],
            }
        else:
            try:
                from tdc.single_pred import ADME

                data = ADME(name=self.dataset_name)
                self.splits = data.get_split(method=self.split_type, seed=self.seed)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load TDC dataset '{self.dataset_name}'. "
                    f"Ensure PyTDC is installed and network is available, or provide synthetic_df. Error: {e}"
                )
        self.is_prepared = True

    def _build_dataset(self, df: pd.DataFrame) -> MolecularDataset:
        samples: List[Dict[str, Any]] = []
        smiles_col = "Drug" if "Drug" in df.columns else "smiles"
        label_col = "Y" if "Y" in df.columns else "label"

        for _, row in df.iterrows():
            s = str(row[smiles_col])
            g = self.graph_transform(s)
            if g is not None:
                lbl = float(row[label_col]) if label_col in row else 0.0
                samples.append({"drug_graph": g, "label": lbl})
        return MolecularDataset(samples)

    def setup_loaders(
        self, batch_size: int = 32, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        self.check_prepared()
        train_ds = self._build_dataset(self.splits["train"])
        val_ds = self._build_dataset(self.splits["valid"])
        test_ds = self._build_dataset(self.splits["test"])

        return (
            DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=molecule_collate_fn,
            ),
            DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=molecule_collate_fn,
            ),
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=molecule_collate_fn,
            ),
        )
