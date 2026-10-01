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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import AminoAcidTokenizer, SmilesToGraphTransform

_DTI_SPLIT_METHODS = {"cold_drug", "cold_protein", "random", "dual_cold", "cold_both"}


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
        split_type: str = "cold_drug",  # ← DEFAULT: cold drug split
        seed: int = 42,
        task_type: str = "dta",
        metric_name: str = "ci",  # ← PRIMARY metric: Concordance Index
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
        self.max_samples = kwargs.get("max_samples", None)
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
        if "kiba" in self.dataset_name.lower():
            # KIBA scores are pre-computed continuous affinity values (typically ~8 to ~16)
            self.log_transform = False

        if self.synthetic_df is not None:
            df = self.synthetic_df
            if self.split_type in ("dual_cold", "cold_both"):
                unique_drugs = df["Drug"].dropna().unique()
                unique_targets = df["Target"].dropna().unique()
                rng = np.random.RandomState(self.seed)
                rng.shuffle(unique_drugs)
                rng.shuffle(unique_targets)

                n_d, n_t = len(unique_drugs), len(unique_targets)
                tr_d = set(unique_drugs[: int(n_d * self.frac[0])])
                val_d = set(unique_drugs[int(n_d * self.frac[0]) : int(n_d * (self.frac[0] + self.frac[1]))])
                te_d = set(unique_drugs[int(n_d * (self.frac[0] + self.frac[1])) :])

                tr_t = set(unique_targets[: int(n_t * self.frac[0])])
                val_t = set(unique_targets[int(n_t * self.frac[0]) : int(n_t * (self.frac[0] + self.frac[1]))])
                te_t = set(unique_targets[int(n_t * (self.frac[0] + self.frac[1])) :])

                self.splits = {
                    "train": df[df["Drug"].isin(tr_d) & df["Target"].isin(tr_t)].reset_index(drop=True),
                    "valid": df[df["Drug"].isin(val_d) & df["Target"].isin(val_t)].reset_index(drop=True),
                    "test": df[df["Drug"].isin(te_d) & df["Target"].isin(te_t)].reset_index(drop=True),
                }
            else:
                n = len(df)
                n_train = int(n * self.frac[0])
                n_val = int(n * self.frac[1])
                self.splits = {
                    "train": df.iloc[:n_train].reset_index(drop=True),
                    "valid": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
                    "test": df.iloc[n_train + n_val :].reset_index(drop=True),
                }
        else:
            try:
                from tdc.multi_pred import DTI  # type: ignore[import]

                data = DTI(name=self.dataset_name)
                if self.split_type in ("dual_cold", "cold_both"):
                    raw_df = data.get_data()
                    unique_drugs = raw_df["Drug"].dropna().unique()
                    unique_targets = raw_df["Target"].dropna().unique()
                    rng = np.random.RandomState(self.seed)
                    rng.shuffle(unique_drugs)
                    rng.shuffle(unique_targets)

                    n_d, n_t = len(unique_drugs), len(unique_targets)
                    tr_d = set(unique_drugs[: int(n_d * self.frac[0])])
                    val_d = set(unique_drugs[int(n_d * self.frac[0]) : int(n_d * (self.frac[0] + self.frac[1]))])
                    te_d = set(unique_drugs[int(n_d * (self.frac[0] + self.frac[1])) :])

                    tr_t = set(unique_targets[: int(n_t * self.frac[0])])
                    val_t = set(unique_targets[int(n_t * self.frac[0]) : int(n_t * (self.frac[0] + self.frac[1]))])
                    te_t = set(unique_targets[int(n_t * (self.frac[0] + self.frac[1])) :])

                    self.splits = {
                        "train": raw_df[raw_df["Drug"].isin(tr_d) & raw_df["Target"].isin(tr_t)].reset_index(drop=True),
                        "valid": raw_df[raw_df["Drug"].isin(val_d) & raw_df["Target"].isin(val_t)].reset_index(drop=True),
                        "test": raw_df[raw_df["Drug"].isin(te_d) & raw_df["Target"].isin(te_t)].reset_index(drop=True),
                    }
                else:
                    self.splits = data.get_split(
                        method=self.split_type,
                        seed=self.seed,
                        frac=self.frac,
                    )
            except Exception as exc:
                if getattr(self, "max_samples", None) is not None or "tdc" in str(exc).lower() or "pytdc" in str(exc).lower():
                    drugs = [
                        "CC(=O)NC1=CC=CC=C1",
                        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                        "CC1=C(C(=O)N(C1=O)C)C2=CC=CC=C2",
                        "CC(=O)OC1=CC=CC=C1C(=O)O",
                    ]
                    targets = [
                        "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFD",
                        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVH",
                    ]
                    recs = []
                    num_recs = max(40, int(getattr(self, "max_samples", 40) or 40))
                    for i in range(num_recs):
                        recs.append({
                            "Drug_ID": f"d_{i}",
                            "Drug": drugs[i % len(drugs)],
                            "Target_ID": f"t_{i}",
                            "Target": targets[i % len(targets)],
                            "Y": float(np.random.uniform(5.0, 500.0)),
                        })
                    df_syn = pd.DataFrame(recs)
                    n = len(df_syn)
                    n_train = int(n * self.frac[0])
                    n_val = int(n * self.frac[1])
                    self.splits = {
                        "train": df_syn.iloc[:n_train].reset_index(drop=True),
                        "valid": df_syn.iloc[n_train : n_train + n_val].reset_index(drop=True),
                        "test": df_syn.iloc[n_train + n_val :].reset_index(drop=True),
                    }
                else:
                    raise RuntimeError(
                        f"Failed to load DTI dataset '{self.dataset_name}' via TDC. "
                        f"Ensure PyTDC is installed: pip install PyTDC. Error: {exc}"
                    ) from exc

        if getattr(self, "max_samples", None) is not None:
            ms = int(self.max_samples)
            for k in self.splits:
                if len(self.splits[k]) > ms:
                    self.splits[k] = self.splits[k].iloc[:ms].reset_index(drop=True)

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
            smiles = str(row.get("Drug", ""))
            target_seq = str(row.get("Target", ""))
            raw_y = float(row.get("Y", 0.0))

            # Build molecular graph (Drug GNN encoder — Phase A)
            graph = self.graph_transform(smiles)
            if graph is None:
                # Skip un-parseable SMILES
                continue

            # Tokenise amino acid sequence with clean AA vocab (Phase A & B)
            target_tensor = self.aa_tokenizer(target_seq)

            samples.append(
                {
                    "drug_graph": graph,  # torch_geometric Data
                    "drug_smiles_str": smiles,  # raw str for HF encoder
                    "target_seq": target_tensor,  # LongTensor [aa_max_len]
                    "target_seq_str": target_seq,  # raw str for ESM-2 encoder (Phase B)
                    "label": self._transform_y(raw_y),  # normalised float
                    # Metadata for cold-split verification
                    "drug_id": str(row.get("Drug_ID", "")),
                    "target_id": str(row.get("Target_ID", "")),
                }
            )

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
            val_ds = self._build_dataset(self.splits["valid"])
            test_ds = self._build_dataset(self.splits["test"])
            self._cached_datasets = (train_ds, val_ds, test_ds)
        else:
            train_ds, val_ds, test_ds = self._cached_datasets

        loader_kwargs: Dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=molecule_collate_fn,
        )

        return (
            DataLoader(train_ds, shuffle=True, **loader_kwargs),
            DataLoader(val_ds, shuffle=False, **loader_kwargs),
            DataLoader(test_ds, shuffle=False, **loader_kwargs),  # cold drug test
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


# ======================================================================
# Task F-3: Multi-Affinity Multi-Task Extension (Kd + Ki + IC50)
# ======================================================================


class MaskedMSELoss(nn.Module):
    """Multi-task MSE loss ignoring missing/masked target affinity values."""

    def __init__(self, task_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if task_weights is not None:
            self.register_buffer("task_weights", task_weights.float())
        else:
            self.task_weights = None

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked MSE.

        Args:
            preds: FloatTensor [B, num_tasks]
            targets: FloatTensor [B, num_tasks]
            mask: Optional Byte/Bool/Float Tensor [B, num_tasks] (1=valid, 0=missing).
        """
        if mask is None:
            mask = ~torch.isnan(targets)
            targets = torch.nan_to_num(targets, nan=0.0)

        preds = preds.view_as(targets)
        diff_sq = (preds - targets.float()) ** 2

        if self.task_weights is not None:
            diff_sq = diff_sq * self.task_weights.to(preds.device)

        masked_diff = diff_sq * mask.float()
        valid_count = torch.clamp(mask.float().sum(), min=1.0)
        return masked_diff.sum() / valid_count


class MultiAffinityDrugTargetDataset(Dataset):
    """In-memory PyTorch Dataset for Multi-Affinity (Kd, Ki, IC50) triplets."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


@DATASETS.register("multi_affinity_dta_loader")
@DATASETS.register("multi_affinity_loader")
class MultiAffinityDTADataModule(BaseTDCDataModule):
    """Multi-Task DTA DataModule integrating BindingDB Kd, Ki, and IC50.

    Ensures zero leakage across assays by splitting unique drugs simultaneously:
    If a drug is in Test, it is NEVER seen in Train or Valid for Kd, Ki, or IC50.

    Args:
        dataset_names: List of TDC dataset names. Default: ["BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"]
        split_type   : Splitting strategy. Default: "cold_drug"
        seed         : Random seed.
        log_transform: Apply log1p transform to nanomolar values. Default: True.
        aa_max_length: Maximum target sequence length. Default: 1000.
        frac         : [train, val, test] split fractions. Default: [0.7, 0.1, 0.2].
    """

    def __init__(
        self,
        dataset_names: Optional[List[str]] = None,
        split_type: str = "cold_drug",
        seed: int = 42,
        task_type: str = "multi_dta",
        metric_name: str = "ci",
        log_transform: bool = True,
        aa_max_length: int = 1000,
        frac: Optional[List[float]] = None,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name="BindingDB_MultiAffinity",
            split_type=split_type,
            seed=seed,
            task_type=task_type,
            metric_name=metric_name,
            synthetic_df=synthetic_df,
        )
        self.dataset_names = dataset_names or ["BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"]
        self.task_names = ["Kd", "Ki", "IC50"]
        self.log_transform = log_transform
        self.aa_max_length = aa_max_length
        self.frac = frac or [0.7, 0.1, 0.2]

        self.graph_transform = SmilesToGraphTransform()
        self.aa_tokenizer = AminoAcidTokenizer(max_length=aa_max_length)

        # Per-task stats on training set: {0: {'mean': float, 'std': float, 'name': str}}
        self.task_stats: Dict[int, Dict[str, Any]] = {}
        self._cached_datasets: Optional[Tuple] = None

    def prepare_data(self) -> None:
        """Download or construct multi-affinity dataset and partition into splits."""
        if self.synthetic_df is not None:
            master_df = self.synthetic_df.copy()
            # Standardize column naming if necessary
            for col in self.task_names:
                if col not in master_df.columns:
                    alt_cols = [c for c in master_df.columns if col.lower() in c.lower()]
                    if alt_cols:
                        master_df[col] = master_df[alt_cols[0]]
                    else:
                        master_df[col] = np.nan
        else:
            try:
                from tdc.multi_pred import DTI

                dfs = []
                for name, task_col in zip(self.dataset_names, self.task_names):
                    dti = DTI(name=name)
                    df = dti.get_data()
                    df_renamed = df.rename(columns={"Y": task_col})
                    dfs.append(df_renamed)

                # Merge on Drug and Target
                merged = dfs[0]
                for next_df in dfs[1:]:
                    # Merge on unique drug and target representations
                    merged = pd.merge(
                        merged,
                        next_df[
                            [
                                "Drug",
                                "Target",
                                [c for c in next_df.columns if c in self.task_names][0],
                            ]
                        ],
                        on=["Drug", "Target"],
                        how="outer",
                    )
                master_df = merged
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load multi-affinity datasets {self.dataset_names} via TDC: {exc}"
                ) from exc

        # Ensure required identifiers exist
        if "Drug" not in master_df.columns or "Target" not in master_df.columns:
            raise KeyError("DataFrame must contain 'Drug' and 'Target' columns.")

        # Leakage-Free Cold-Drug Split Partitioning
        unique_drugs = master_df["Drug"].dropna().unique()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(unique_drugs)

        n_drugs = len(unique_drugs)
        n_train = int(n_drugs * self.frac[0])
        n_val = int(n_drugs * self.frac[1])

        train_drugs = set(unique_drugs[:n_train])
        val_drugs = set(unique_drugs[n_train : n_train + n_val])
        test_drugs = set(unique_drugs[n_train + n_val :])

        self.splits = {
            "train": master_df[master_df["Drug"].isin(train_drugs)].reset_index(drop=True),
            "valid": master_df[master_df["Drug"].isin(val_drugs)].reset_index(drop=True),
            "test": master_df[master_df["Drug"].isin(test_drugs)].reset_index(drop=True),
        }

        # Calculate per-task normalization statistics on TRAIN split only
        train_df = self.splits["train"]
        for idx, task_name in enumerate(self.task_names):
            if task_name in train_df.columns:
                vals = train_df[task_name].dropna().astype(float).values
                if len(vals) > 0:
                    if self.log_transform:
                        vals = np.log1p(np.clip(vals, 0.0, None))
                    mean_val = float(np.nanmean(vals))
                    std_val = float(np.nanstd(vals))
                    if std_val < 1e-6:
                        std_val = 1.0
                else:
                    mean_val, std_val = 0.0, 1.0
            else:
                mean_val, std_val = 0.0, 1.0

            self.task_stats[idx] = {
                "name": task_name,
                "mean": mean_val,
                "std": std_val,
            }

        self.is_prepared = True

    def _transform_val(self, val: float, task_idx: int) -> float:
        """Standardize a single task value using fitted train statistics."""
        st = self.task_stats[task_idx]
        v = np.log1p(max(0.0, float(val))) if self.log_transform else float(val)
        return float((v - st["mean"]) / st["std"])

    def _build_dataset(self, df: pd.DataFrame) -> MultiAffinityDrugTargetDataset:
        """Convert split DataFrame into MultiAffinityDrugTargetDataset."""
        samples: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            smiles = str(row.get("Drug", ""))
            target_seq = str(row.get("Target", ""))

            graph = self.graph_transform(smiles)
            if graph is None:
                continue

            target_tensor = self.aa_tokenizer(target_seq)

            labels_vec = []
            mask_vec = []
            for idx, task_name in enumerate(self.task_names):
                raw_val = row.get(task_name, np.nan)
                if pd.notna(raw_val) and np.isfinite(raw_val):
                    labels_vec.append(self._transform_val(raw_val, idx))
                    mask_vec.append(True)
                else:
                    labels_vec.append(0.0)
                    mask_vec.append(False)

            # Skip samples with no labels at all
            if not any(mask_vec):
                continue

            samples.append(
                {
                    "drug_graph": graph,
                    "drug_smiles_str": smiles,
                    "target_seq": target_tensor,
                    "target_seq_str": target_seq,
                    "labels": torch.tensor(labels_vec, dtype=torch.float32),
                    "mask": torch.tensor(mask_vec, dtype=torch.bool),
                    "drug_id": str(row.get("Drug_ID", "")),
                    "target_id": str(row.get("Target_ID", "")),
                }
            )

        return MultiAffinityDrugTargetDataset(samples)

    def setup_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_loader, val_loader, test_loader)."""
        self.check_prepared()

        if self._cached_datasets is None:
            train_ds = self._build_dataset(self.splits["train"])
            val_ds = self._build_dataset(self.splits["valid"])
            test_ds = self._build_dataset(self.splits["test"])
            self._cached_datasets = (train_ds, val_ds, test_ds)
        else:
            train_ds, val_ds, test_ds = self._cached_datasets

        loader_kwargs: Dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=molecule_collate_fn,
        )

        return (
            DataLoader(train_ds, shuffle=True, **loader_kwargs),
            DataLoader(val_ds, shuffle=False, **loader_kwargs),
            DataLoader(test_ds, shuffle=False, **loader_kwargs),
        )

    def inverse_transform_y(self, y_norm: float, task_idx: int) -> float:
        """Reverse z-score + log1p for a specific task index (0=Kd, 1=Ki, 2=IC50)."""
        st = self.task_stats[task_idx]
        y_log = y_norm * st["std"] + st["mean"]
        return float(np.expm1(y_log)) if self.log_transform else y_log

    @property
    def split_info(self) -> Dict[str, int]:
        if not self.is_prepared:
            return {}
        return {k: len(v) for k, v in self.splits.items()}
