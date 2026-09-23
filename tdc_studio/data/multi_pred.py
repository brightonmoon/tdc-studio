"""Multi-prediction dataset loaders for Drug-Target Interaction (DTI/DTA) tasks.

Design principles:
- cold_drug split is the DEFAULT to reflect real new-drug-discovery scenarios.
- Each batch item contains both drug_graph (for GNN) and drug_smiles_str
  (for future HuggingFace encoder swap in Phase B) alongside target_seq.
- Uses AminoAcidTokenizer (not SequenceTokenizer) to avoid SMILES vocab cross-pollution.
- Applies log1p transform on affinity values to reduce dynamic range skew
  (Kd in nM can span 6+ orders of magnitude).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import AminoAcidTokenizer, SmilesToGraphTransform

# Valid split methods for DTI (TDC-supported)
_DTI_SPLIT_METHODS = {"cold_drug", "cold_protein", "random"}


class DrugTargetPairDataset(Dataset):
    """In-memory PyTorch Dataset for (Drug, Target, Affinity) triplets.

    Each item contains:
        drug_graph     : torch_geometric.data.Data  — for GNN-based encoders (Phase A)
        drug_smiles_str: str                        — for HuggingFace encoders (Phase B)
        target_seq     : torch.LongTensor [max_len] — tokenised AA sequence
        label          : float                      — normalised affinity value
        drug_id        : str                        — TDC drug identifier (for split audit)
        target_id      : str                        — TDC target identifier
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


@DATASETS.register("dta_loader")
@DATASETS.register("dti_loader")
class DTADataModule(BaseTDCDataModule):
    """Data module for Drug-Target Interaction / Affinity prediction.

    Wraps the TDC multi-prediction DTI API with opinionated defaults for
    new-drug discovery evaluation:
    - Defaults to cold_drug split (not random) to evaluate new-drug generalisation.
    - Provides both drug_graph and drug_smiles_str so models can choose between
      GNN-based (Phase A) and pretrained LM-based (Phase B) encoders without
      changing the DataModule.
    - Applies log1p + z-score normalisation to affinity values.

    Args:
        dataset_name : TDC DTI dataset name (e.g. 'BindingDB_Kd', 'DAVIS', 'KIBA').
        split_type   : Splitting strategy. Defaults to 'cold_drug'.
                       Options: 'cold_drug', 'cold_protein', 'random'.
        seed         : Random seed for reproducibility.
        log_transform: Apply log1p(y) transform to affinity values. Default True.
        aa_max_length: Maximum amino acid sequence length for truncation/padding.
        frac         : Train/val/test fraction list. Default [0.7, 0.1, 0.2].
    """

    def __init__(
        self,
        dataset_name: str = "BindingDB_Kd",
        split_type: str = "cold_drug",           # ← DEFAULT: cold drug split
        seed: int = 42,
        task_type: str = "dta",
        metric_name: str = "ci",                 # ← PRIMARY metric: Concordance Index
        log_transform: bool = True,
        aa_max_length: int = 1000,
        frac: Optional[List[float]] = None,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        if split_type not in _DTI_SPLIT_METHODS:
            raise ValueError(
                f"split_type='{split_type}' is not valid for DTI. "
                f"Choose from: {sorted(_DTI_SPLIT_METHODS)}"
            )
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type=task_type,
            metric_name=metric_name,
            synthetic_df=synthetic_df,
        )
        self.log_transform = log_transform
        self.aa_max_length = aa_max_length
        self.frac = frac or [0.7, 0.1, 0.2]

        # Transforms — AminoAcidTokenizer uses clean 25-token AA vocab
        self.graph_transform = SmilesToGraphTransform()
        self.aa_tokenizer = AminoAcidTokenizer(max_length=aa_max_length)

        # Affinity statistics (fitted on train split after prepare_data)
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self._cached_datasets: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download and split the DTI dataset via TDC API.

        Uses TDC cold_drug split by default. Fits affinity normalisation
        statistics on the training split only (no data leakage from val/test).
        """
        if self.synthetic_df is not None:
            df = self.synthetic_df
            n = len(df)
            n_train = int(n * self.frac[0])
            n_val = int(n * self.frac[1])
            self.splits = {
                "train": df.iloc[:n_train].reset_index(drop=True),
                "valid": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
                "test":  df.iloc[n_train + n_val :].reset_index(drop=True),
            }
        else:
            try:
                from tdc.multi_pred import DTI  # type: ignore[import]

                data = DTI(name=self.dataset_name)
                self.splits = data.get_split(
                    method=self.split_type,
                    seed=self.seed,
                    frac=self.frac,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load DTI dataset '{self.dataset_name}' via TDC. "
                    f"Ensure PyTDC is installed: pip install PyTDC. Error: {exc}"
                ) from exc

        # Fit log+standardisation scaler on train split ONLY
        train_y = self.splits["train"]["Y"].astype(float).values
        if self.log_transform:
            train_y = np.log1p(np.clip(train_y, 0.0, None))
        self.y_mean = float(np.nanmean(train_y))
        self.y_std = float(np.nanstd(train_y))
        if self.y_std < 1e-6:
            self.y_std = 1.0

        self.is_prepared = True

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def _transform_y(self, y: float) -> float:
        """Apply log1p + z-score normalisation to a raw affinity value."""
        val = np.log1p(max(0.0, float(y))) if self.log_transform else float(y)
        return float((val - self.y_mean) / self.y_std)

    def _build_dataset(self, df: pd.DataFrame) -> DrugTargetPairDataset:
        """Convert a TDC split DataFrame into a DrugTargetPairDataset.

        Expected TDC DTI DataFrame columns:
            Drug_ID | Drug (SMILES) | Target_ID | Target (AA seq) | Y
        """
        samples: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            smiles     = str(row.get("Drug", ""))
            target_seq = str(row.get("Target", ""))
            raw_y      = float(row.get("Y", 0.0))

            # Build molecular graph (Drug GNN encoder — Phase A)
            graph = self.graph_transform(smiles)
            if graph is None:
                # Skip un-parseable SMILES
                continue

            # Tokenise amino acid sequence with clean AA vocab (Phase A & B)
            target_tensor = self.aa_tokenizer(target_seq)

            samples.append({
                "drug_graph":      graph,                    # torch_geometric Data
                "drug_smiles_str": smiles,                   # raw str for HF encoder
                "target_seq":      target_tensor,            # LongTensor [aa_max_len]
                "label":           self._transform_y(raw_y), # normalised float
                # Metadata for cold-split verification
                "drug_id":   str(row.get("Drug_ID", "")),
                "target_id": str(row.get("Target_ID", "")),
            })

        return DrugTargetPairDataset(samples)

    # ------------------------------------------------------------------
    # DataLoader setup
    # ------------------------------------------------------------------

    def setup_loaders(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_loader, val_loader, test_loader).

        test_loader drugs are entirely unseen in train by default (cold_drug),
        which is the evaluation that matters for new-drug-discovery applications.
        """
        self.check_prepared()

        if self._cached_datasets is None:
            train_ds = self._build_dataset(self.splits["train"])
            val_ds   = self._build_dataset(self.splits["valid"])
            test_ds  = self._build_dataset(self.splits["test"])
            self._cached_datasets = (train_ds, val_ds, test_ds)
        else:
            train_ds, val_ds, test_ds = self._cached_datasets

        loader_kwargs: Dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=molecule_collate_fn,
        )

        return (
            DataLoader(train_ds, shuffle=True,  **loader_kwargs),
            DataLoader(val_ds,   shuffle=False, **loader_kwargs),
            DataLoader(test_ds,  shuffle=False, **loader_kwargs),  # cold drug test
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def inverse_transform_y(self, y_norm: float) -> float:
        """Reverse z-score + log1p to recover original affinity scale (nM)."""
        y_log = y_norm * self.y_std + self.y_mean
        return float(np.expm1(y_log)) if self.log_transform else y_log

    @property
    def split_info(self) -> Dict[str, int]:
        """Number of (Drug, Target) pairs in each split."""
        if not self.is_prepared:
            return {}
        return {k: len(v) for k, v in self.splits.items()}
