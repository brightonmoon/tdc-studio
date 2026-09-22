"""Bio-Permeability Multi-Task Learning Data Module.

Combines Caco-2 Wang permeability with sister biophysical endpoints:
- lipophilicity_astrazeneca (logD 7.4): passive lipid membrane partition driver
- solubility_aqsoldb (logS): aqueous dissolution bottleneck
- hia_hou: human intestinal absorption in vivo counterpart

Strictly preserves official TDC Caco-2 scaffold benchmark partition:
zero test set leakage into auxiliary training splits.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule, MolecularDataset
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.single_pred import _load_tdc_fallback
from tdc_studio.data.transforms import (
    MorganFingerprintTransform,
    RDKit2DDescriptorsTransform,
    SmilesToGraphTransform,
    SmilesTokenizer,
)
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss

BIO_PERMEABILITY_TASKS: List[Dict[str, str]] = [
    {"name": "caco2_wang", "category": "absorption", "type": "regression"},
    {"name": "lipophilicity_astrazeneca", "category": "physicochemical", "type": "regression"},
    {"name": "solubility_aqsoldb", "category": "physicochemical", "type": "regression"},
    {"name": "hia_hou", "category": "absorption", "type": "binary_classification"},
]


def _canonicalize_smiles(s: str) -> Optional[str]:
    """Convert SMILES to canonical RDKit SMILES for exact cross-dataset matching."""
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        pass
    return None


@DATASETS.register("bio_permeability_loader")
@DATASETS.register("bio_permeability_mtl")
class BioPermeabilityDataModule(BaseTDCDataModule):
    """Multi-task Data Module for Bio-Permeability combining Caco-2, Lipophilicity, Solubility, and HIA."""

    def __init__(
        self,
        tasks: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        dataset_name: str = "bio_permeability_mtl",
        split_type: str = "scaffold",
        seed: int = 42,
        modality: str = "graph",
        primary_task: str = "caco2_wang",
        use_descriptors: bool = True,
        standardize_target: bool = True,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type="multi_task",
            metric_name="composite",
            synthetic_df=synthetic_df,
        )
        self.modality = modality.lower()
        self.primary_task = primary_task
        self.use_descriptors = use_descriptors
        self.standardize_target = standardize_target

        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()
        self.desc_transform = RDKit2DDescriptorsTransform() if self.use_descriptors else None

        # Parse task list (defaults to BIO_PERMEABILITY_TASKS)
        raw_tasks = tasks if tasks is not None else BIO_PERMEABILITY_TASKS
        self.task_configs: List[Dict[str, str]] = []
        for t in raw_tasks:
            if isinstance(t, str):
                self.task_configs.append(
                    {
                        "name": t,
                        "category": "absorption" if "caco" in t or "hia" in t else "physicochemical",
                        "type": "binary_classification" if "hia" in t else "regression",
                    }
                )
            elif isinstance(t, dict):
                self.task_configs.append(
                    {
                        "name": t["name"],
                        "category": t.get("category", "absorption"),
                        "type": t.get("type", "regression"),
                    }
                )

        self.task_names = [t["name"] for t in self.task_configs]
        self.task_types = [t["type"] for t in self.task_configs]
        self.num_tasks = len(self.task_configs)
        self.task_stats: Dict[str, Dict[str, float]] = {}

    def get_loss_fn(
        self,
        use_uncertainty: bool = False,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> MaskedMultiTaskLoss:
        """Construct MaskedMultiTaskLoss configured for Bio-Permeability."""
        # Default weighting: emphasize primary Caco-2 task
        weights = task_weights or {
            "caco2_wang": 2.0,
            "lipophilicity_astrazeneca": 0.5,
            "solubility_aqsoldb": 0.5,
            "hia_hou": 0.5,
        }
        return MaskedMultiTaskLoss(
            task_names=self.task_names,
            task_types=self.task_types,
            use_uncertainty=use_uncertainty,
            task_weights=weights,
        )

    def prepare_data(self) -> None:
        """Prepare train/valid/test splits preserving Caco-2 benchmark scaffold split."""
        if self.synthetic_df is not None:
            df = self.synthetic_df.copy()
            n = len(df)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            self.splits = {
                "train": df.iloc[:n_train].reset_index(drop=True),
                "valid": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
                "test": df.iloc[n_train + n_val :].reset_index(drop=True),
            }
        else:
            self._load_and_merge_real_datasets()

        # Compute training mean and std for regression tasks
        train_df = self.splits["train"]
        for t_name, t_type in zip(self.task_names, self.task_types):
            if t_type == "regression" and self.standardize_target and t_name in train_df.columns:
                valid_vals = train_df[t_name].dropna().values
                if len(valid_vals) > 0:
                    mean_val = float(np.mean(valid_vals))
                    std_val = float(np.std(valid_vals))
                    if std_val < 1e-6:
                        std_val = 1.0
                    self.task_stats[t_name] = {"mean": mean_val, "std": std_val}
                else:
                    self.task_stats[t_name] = {"mean": 0.0, "std": 1.0}
            else:
                self.task_stats[t_name] = {"mean": 0.0, "std": 1.0}

        self.is_prepared = True

    def _load_and_merge_real_datasets(self) -> None:
        """Fetch TDC datasets with Caco-2 scaffold split as primary benchmark anchor."""
        caco2_splits = None
        try:
            from tdc.single_pred import ADME

            caco2_data = ADME(name="caco2_wang")
            caco2_splits = caco2_data.get_split(method=self.split_type, seed=self.seed)
        except Exception:
            caco2_splits = _load_tdc_fallback(
                dataset_name="caco2_wang", split_type=self.split_type, seed=self.seed
            )

        caco2_train = caco2_splits["train"][["Drug", "Y"]].rename(columns={"Y": "caco2_wang"})
        caco2_valid = caco2_splits["valid"][["Drug", "Y"]].rename(columns={"Y": "caco2_wang"})
        caco2_test = caco2_splits["test"][["Drug", "Y"]].rename(columns={"Y": "caco2_wang"})

        # Canonicalize Caco-2 SMILES for exact sets
        caco2_train["Canon_SMILES"] = caco2_train["Drug"].apply(_canonicalize_smiles)
        caco2_valid["Canon_SMILES"] = caco2_valid["Drug"].apply(_canonicalize_smiles)
        caco2_test["Canon_SMILES"] = caco2_test["Drug"].apply(_canonicalize_smiles)

        train_smiles_set = set(caco2_train["Canon_SMILES"].dropna())
        valid_smiles_set = set(caco2_valid["Canon_SMILES"].dropna())
        test_smiles_set = set(caco2_test["Canon_SMILES"].dropna())

        # Load auxiliary datasets
        aux_data_dict: Dict[str, pd.DataFrame] = {}
        for t_name in self.task_names:
            if t_name == "caco2_wang":
                continue
            try:
                from tdc.single_pred import ADME

                aux_d = ADME(name=t_name).get_data()
                sub = aux_d[["Drug", "Y"]].rename(columns={"Y": t_name})
                sub["Canon_SMILES"] = sub["Drug"].apply(_canonicalize_smiles)
                aux_data_dict[t_name] = sub.dropna(subset=["Canon_SMILES"])
            except Exception:
                # If auxiliary dataset fails, proceed with available tasks
                continue

        # Partition auxiliary compounds strictly respecting Caco-2 benchmark
        aux_train_list = [caco2_train]
        aux_valid_list = [caco2_valid]
        aux_test_list = [caco2_test]

        for t_name, aux_df in aux_data_dict.items():
            # Never leak test smiles into train or valid
            t_test = aux_df[aux_df["Canon_SMILES"].isin(test_smiles_set)]
            t_val = aux_df[aux_df["Canon_SMILES"].isin(valid_smiles_set)]
            t_train = aux_df[aux_df["Canon_SMILES"].isin(train_smiles_set)]

            # Novel compounds not in Caco-2: split 85% train, 15% valid
            novel = aux_df[
                ~aux_df["Canon_SMILES"].isin(test_smiles_set)
                & ~aux_df["Canon_SMILES"].isin(valid_smiles_set)
                & ~aux_df["Canon_SMILES"].isin(train_smiles_set)
            ].sample(frac=1.0, random_state=self.seed)

            n_nov = len(novel)
            n_tr = int(n_nov * 0.85)
            nov_train = novel.iloc[:n_tr]
            nov_val = novel.iloc[n_tr:]

            aux_train_list.append(pd.concat([t_train, nov_train], ignore_index=True))
            aux_valid_list.append(pd.concat([t_val, nov_val], ignore_index=True))
            if len(t_test) > 0:
                aux_test_list.append(t_test)

        # Merge partitions by Canon_SMILES
        def _merge_partition(dfs: List[pd.DataFrame]) -> pd.DataFrame:
            res = dfs[0]
            for other in dfs[1:]:
                res = pd.merge(res, other, on=["Canon_SMILES", "Drug"], how="outer")
            return res.drop_duplicates(subset=["Canon_SMILES"]).reset_index(drop=True)

        self.splits = {
            "train": _merge_partition(aux_train_list),
            "valid": _merge_partition(aux_valid_list),
            "test": _merge_partition(aux_test_list),
        }

    def _build_dataset(self, df: pd.DataFrame) -> MolecularDataset:
        samples: List[Dict[str, Any]] = []
        smiles_col = "Drug" if "Drug" in df.columns else "smiles"

        for _, row in df.iterrows():
            s = str(row[smiles_col])
            labels: List[float] = []
            mask: List[bool] = []

            for t_name, t_type in zip(self.task_names, self.task_types):
                if t_name in row and pd.notna(row[t_name]):
                    val = float(row[t_name])
                    if t_type == "regression" and self.standardize_target:
                        stat = self.task_stats.get(t_name, {"mean": 0.0, "std": 1.0})
                        val = (val - stat["mean"]) / stat["std"]
                    labels.append(val)
                    mask.append(True)
                else:
                    labels.append(0.0)
                    mask.append(False)

            labels_t = torch.tensor(labels, dtype=torch.float32)
            mask_t = torch.tensor(mask, dtype=torch.bool)
            sample: Dict[str, Any] = {"labels": labels_t, "mask": mask_t}

            if self.modality == "graph":
                g = self.graph_transform(s)
                if g is not None:
                    sample["drug_graph"] = g
                    if self.use_descriptors and self.desc_transform is not None:
                        desc = self.desc_transform(s)
                        if desc is not None:
                            sample["descriptors"] = desc
                    samples.append(sample)
            elif self.modality == "sequence":
                seq = self.smiles_tokenizer(s)
                sample["smiles_seq"] = seq
                samples.append(sample)
            elif self.modality == "fingerprint":
                fp = self.fingerprint_transform(s)
                if fp is not None:
                    sample["fingerprint"] = fp
                    samples.append(sample)

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
