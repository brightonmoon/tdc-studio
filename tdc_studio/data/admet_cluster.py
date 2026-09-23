"""ADMET Multi-Cluster Data Module.

Provides unified multi-task dataset loading for all ADMET clusters:
- Cluster 1: Lipophilicity & Permeability (Lipophilicity, Solubility, FreeSolv, Caco-2)
- Cluster 2: Plasma Distribution (PPBR, BBB, VDss, Lipophilicity)
- Cluster 3: CYP450 Metabolism Matrix (5 Inhibitors + 3 Substrates)
- Cluster 4: Clearance & Elimination (Half-Life, Hepatocyte CL, Microsome CL, CYP3A4, PPBR)
- Cluster 5: Toxicity Profile (hERG, LD50, DILI, AMES, Skin Reaction, Carcinogens)
- Specialized: Standalone Cardiotoxicity (hERG Central 306k 3-head joint learning)

Enforces strict scaffold benchmark isolation of the primary target:
auxiliary task molecules are filtered such that primary benchmark test compounds
never leak into training or validation splits.
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

KNOWN_TOX_DATASETS = {
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
    "skin_reaction",
    "carcinogens_lagunin",
    "clintox",
    "herg_karim",
    "herg_central",
}


def _canonicalize_smiles(s: Any) -> Optional[str]:
    """Convert SMILES to canonical RDKit SMILES for exact cross-dataset alignment."""
    if pd.isna(s):
        return None
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        pass
    return None


def _fetch_tdc_dataset(name: str, split_type: str = "scaffold", seed: int = 42, get_split: bool = False) -> Any:
    """Fetch TDC dataset from ADME or Tox with automated fallback or local external cache."""
    from pathlib import Path
    if name.lower() == "chembl_hsa":
        p = Path("data/external/chembl_hsa_processed.csv")
        if p.exists():
            return pd.read_csv(p)

    from tdc.single_pred import ADME, Tox

    is_tox = name.lower() in KNOWN_TOX_DATASETS or "tox" in name.lower()
    primary_cls = Tox if is_tox else ADME
    secondary_cls = ADME if is_tox else Tox

    try:
        loader = primary_cls(name=name)
        return loader.get_split(method=split_type, seed=seed) if get_split else loader.get_data()
    except Exception:
        try:
            loader = secondary_cls(name=name)
            return loader.get_split(method=split_type, seed=seed) if get_split else loader.get_data()
        except Exception:
            return _load_tdc_fallback(dataset_name=name, split_type=split_type, seed=seed)


@DATASETS.register("admet_cluster_loader")
@DATASETS.register("admet_cluster_mtl")
@DATASETS.register("admet_mtl")
class ADMETClusterDataModule(BaseTDCDataModule):
    """Unified Multi-Task Data Module for all ADMET Clusters and hERG Standalone."""

    def __init__(
        self,
        tasks: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        dataset_name: str = "admet_cluster_mtl",
        split_type: str = "scaffold",
        seed: int = 42,
        modality: str = "graph",
        primary_task: Optional[str] = None,
        use_descriptors: bool = True,
        standardize_target: bool = True,
        max_samples: Optional[int] = None,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type="multi_task",
            metric_name=kwargs.get("metric_name", "composite"),
            synthetic_df=synthetic_df,
        )
        self.modality = modality.lower()
        self.use_descriptors = use_descriptors
        self.standardize_target = standardize_target
        self.max_samples = max_samples

        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()
        self.desc_transform = RDKit2DDescriptorsTransform() if self.use_descriptors else None

        # Parse task list
        raw_tasks = tasks or [
            {"name": "caco2_wang", "category": "absorption", "type": "regression"},
            {"name": "lipophilicity_astrazeneca", "category": "physicochemical", "type": "regression"},
            {"name": "solubility_aqsoldb", "category": "physicochemical", "type": "regression"},
            {"name": "hia_hou", "category": "absorption", "type": "binary_classification"},
        ]

        self.task_configs: List[Dict[str, Any]] = []
        for t in raw_tasks:
            if isinstance(t, str):
                is_clf = any(term in t.lower() for term in ("inhib", "substrate", "hia", "bbb", "herg", "ames", "dili", "reaction", "carcinogen", "clintox"))
                self.task_configs.append(
                    {
                        "name": t,
                        "label_name": t,
                        "category": "other",
                        "type": "binary_classification" if is_clf else "regression",
                    }
                )
            elif isinstance(t, dict):
                self.task_configs.append(
                    {
                        "name": t["name"],
                        "label_name": t.get("label_name", t["name"]),
                        "category": t.get("category", "other"),
                        "type": t.get("type", "regression"),
                        "transform": t.get("transform", None),
                    }
                )

        self.task_names = [t["name"] for t in self.task_configs]
        self.task_types = [t["type"] for t in self.task_configs]
        self.task_transforms = {t["name"]: t.get("transform", None) for t in self.task_configs}
        self.num_tasks = len(self.task_configs)
        self.task_stats: Dict[str, Dict[str, float]] = {}

        # Set primary anchor task
        self.primary_task = primary_task or (self.task_names[0] if self.task_names else "primary")

    def get_loss_fn(
        self,
        use_uncertainty: bool = False,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> MaskedMultiTaskLoss:
        """Construct MaskedMultiTaskLoss configured for this cluster."""
        return MaskedMultiTaskLoss(
            task_names=self.task_names,
            task_types=self.task_types,
            use_uncertainty=use_uncertainty,
            task_weights=task_weights,
        )

    def prepare_data(self) -> None:
        """Prepare train/valid/test splits preserving primary benchmark scaffold partition."""
        if self.synthetic_df is not None:
            df = self.synthetic_df.copy()
            n = len(df)
            n_train = max(1, int(n * 0.6))
            n_val = max(1, int(n * 0.2))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            self.splits = {
                "train": df.iloc[:n_train].reset_index(drop=True),
                "valid": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
                "test": df.iloc[n_train + n_val :].reset_index(drop=True),
            }
        elif self.dataset_name.lower() == "herg_central":
            self._load_herg_central()
        else:
            self._load_and_merge_cluster_datasets()

        # Compute training mean and std for regression tasks
        train_df = self.splits["train"]
        for t_name, t_type in zip(self.task_names, self.task_types):
            if t_type == "regression" and self.standardize_target and t_name in train_df.columns:
                valid_vals = train_df[t_name].dropna().values.astype(float)
                if len(valid_vals) > 0:
                    if self.task_transforms.get(t_name) == "logit":
                        fb = np.clip(valid_vals / 100.0, 1e-4, 1.0 - 1e-4)
                        valid_vals = np.log(fb / (1.0 - fb))
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

    def _load_herg_central(self) -> None:
        """Load NIH NCATS hERG Central 306k multi-assay dataset."""
        import os
        from pathlib import Path

        tab_path = Path("data/herg_central.tab")
        if not tab_path.exists():
            from tdc.single_pred import Tox
            _ = Tox(name="herg_central", label_name="hERG_at_1uM")

        df = pd.read_csv(tab_path, sep="\t")
        smiles_col = "X" if "X" in df.columns else ("Drug" if "Drug" in df.columns else df.columns[1])
        df = df.rename(columns={smiles_col: "Drug"})

        # Mapping config tasks (label_name -> task_name)
        col_map = {t["label_name"]: t["name"] for t in self.task_configs if t["label_name"] in df.columns}
        df = df.rename(columns=col_map)

        if self.max_samples is not None:
            df = df.iloc[: max(self.max_samples * 5, 200)].copy()

        df["Canon_SMILES"] = df["Drug"].apply(_canonicalize_smiles)
        df = df.dropna(subset=["Canon_SMILES"]).reset_index(drop=True)

        n = len(df)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        self.splits = {
            "train": df.iloc[:n_train].reset_index(drop=True),
            "valid": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
            "test": df.iloc[n_train + n_val :].reset_index(drop=True),
        }

    def _load_and_merge_cluster_datasets(self) -> None:
        """Fetch TDC datasets with primary task scaffold split as benchmark anchor."""
        primary_task = self.primary_task
        primary_splits = _fetch_tdc_dataset(
            name=primary_task, split_type=self.split_type, seed=self.seed, get_split=True
        )

        def _extract_primary(df: pd.DataFrame) -> pd.DataFrame:
            smiles_col = "Drug" if "Drug" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[1])
            sub = df[[smiles_col, "Y"]].rename(columns={smiles_col: "Drug", "Y": primary_task})
            sub["Canon_SMILES"] = sub["Drug"].apply(_canonicalize_smiles)
            return sub.dropna(subset=["Canon_SMILES"]).reset_index(drop=True)

        p_train = _extract_primary(primary_splits["train"])
        p_valid = _extract_primary(primary_splits["valid"])
        p_test = _extract_primary(primary_splits["test"])

        train_smiles_set = set(p_train["Canon_SMILES"])
        valid_smiles_set = set(p_valid["Canon_SMILES"])
        test_smiles_set = set(p_test["Canon_SMILES"])

        # Load auxiliary datasets
        aux_data_dict: Dict[str, pd.DataFrame] = {}
        for t_cfg in self.task_configs:
            t_name = t_cfg["name"]
            if t_name == primary_task:
                continue
            try:
                aux_d = _fetch_tdc_dataset(name=t_name, get_split=False)
                smiles_col = "Drug" if "Drug" in aux_d.columns else ("smiles" if "smiles" in aux_d.columns else aux_d.columns[1])
                sub = aux_d[[smiles_col, "Y"]].rename(columns={smiles_col: "Drug", "Y": t_name})
                sub["Canon_SMILES"] = sub["Drug"].apply(_canonicalize_smiles)
                aux_data_dict[t_name] = sub.dropna(subset=["Canon_SMILES"])
            except Exception:
                continue

        # Partition auxiliary compounds strictly respecting primary benchmark
        aux_train_list = [p_train]
        aux_valid_list = [p_valid]
        aux_test_list = [p_test]

        for t_name, aux_df in aux_data_dict.items():
            # Zero leakage of primary test smiles into auxiliary train or valid
            t_test = aux_df[aux_df["Canon_SMILES"].isin(test_smiles_set)]
            t_val = aux_df[aux_df["Canon_SMILES"].isin(valid_smiles_set)]
            t_train = aux_df[aux_df["Canon_SMILES"].isin(train_smiles_set)]

            # Novel compounds not present in primary dataset: 85% train, 15% valid, 0% test
            novel = aux_df[
                ~aux_df["Canon_SMILES"].isin(test_smiles_set | valid_smiles_set | train_smiles_set)
            ].sample(frac=1.0, random_state=self.seed)

            n_nov = len(novel)
            n_tr = int(n_nov * 0.85)
            nov_train = novel.iloc[:n_tr]
            nov_val = novel.iloc[n_tr:]

            aux_train_list.append(pd.concat([t_train, nov_train], ignore_index=True))
            aux_valid_list.append(pd.concat([t_val, nov_val], ignore_index=True))
            if len(t_test) > 0:
                aux_test_list.append(t_test)

        # Merge partitions by Canon_SMILES preserving valid Drug and Canon_SMILES
        def _merge_partition(dfs: List[pd.DataFrame]) -> pd.DataFrame:
            if not dfs:
                return pd.DataFrame()
            first = dfs[0].copy()
            res = first[["Canon_SMILES", "Drug", primary_task]]
            for other in dfs[1:]:
                task_cols = [c for c in other.columns if c not in ("Drug", "Canon_SMILES")]
                other_cols = ["Canon_SMILES"] + (["Drug"] if "Drug" in other.columns else []) + task_cols
                merged = pd.merge(res, other[other_cols], on="Canon_SMILES", how="outer", suffixes=("", "_other"))
                if "Drug_other" in merged.columns:
                    merged["Drug"] = merged["Drug"].fillna(merged["Drug_other"])
                    merged = merged.drop(columns=["Drug_other"])
                res = merged
            res["Drug"] = res["Drug"].fillna(res["Canon_SMILES"])
            return res.drop_duplicates(subset=["Canon_SMILES"]).reset_index(drop=True)

        self.splits = {
            "train": _merge_partition(aux_train_list),
            "valid": _merge_partition(aux_valid_list),
            "test": _merge_partition(aux_test_list),
        }

    def _build_dataset(self, df: pd.DataFrame) -> MolecularDataset:
        samples: List[Dict[str, Any]] = []

        target_df = df
        if self.max_samples is not None and len(df) > self.max_samples:
            target_df = df.iloc[: self.max_samples]

        for _, row in target_df.iterrows():
            s = row.get("Canon_SMILES")
            if pd.isna(s) or str(s).lower() == "nan":
                s = row.get("Drug")
            s = str(s) if pd.notna(s) else ""
            if not s or s.lower() == "nan":
                continue
            labels: List[float] = []
            mask: List[bool] = []

            for t_name, t_type in zip(self.task_names, self.task_types):
                if t_name in row and pd.notna(row[t_name]):
                    val = float(row[t_name])
                    if self.task_transforms.get(t_name) == "logit":
                        fb = np.clip(val / 100.0, 1e-4, 1.0 - 1e-4)
                        val = float(np.log(fb / (1.0 - fb)))
                    if self.standardize_target and t_type == "regression":
                        stat = self.task_stats.get(t_name, {"mean": 0.0, "std": 1.0})
                        val = (val - stat["mean"]) / stat["std"]
                    labels.append(val)
                    mask.append(True)
                else:
                    labels.append(0.0)
                    mask.append(False)

            labels_t = torch.tensor(labels, dtype=torch.float32)
            mask_t = torch.tensor(mask, dtype=torch.bool)
            sample: Dict[str, Any] = {"labels": labels_t, "mask": mask_t, "drug_smiles_str": s}

            if self.modality == "graph":
                g = self.graph_transform(s)
                if g is not None:
                    sample["drug_graph"] = g
                    if self.use_descriptors and self.desc_transform:
                        desc = self.desc_transform(s)
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

        # Transparent disk caching for precomputed graph & descriptor datasets
        cache_path = None
        if self.synthetic_df is None and self.max_samples is None:
            import hashlib
            from pathlib import Path
            task_str = "_".join(sorted(self.task_names)) + "_" + "_".join(f"{k}:{v}" for k, v in sorted(self.task_transforms.items()) if v)
            cache_name = f"{self.dataset_name}_{self.primary_task}_{self.split_type}_{self.seed}_{self.modality}_{self.use_descriptors}_{self.standardize_target}_{hashlib.md5(task_str.encode()).hexdigest()[:8]}.pt"
            cache_dir = Path("data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / cache_name

        train_ds, val_ds, test_ds = None, None, None
        if cache_path and cache_path.exists():
            try:
                loaded = torch.load(cache_path, weights_only=False)
                train_ds = loaded.get("train")
                val_ds = loaded.get("val")
                test_ds = loaded.get("test")
            except Exception:
                train_ds, val_ds, test_ds = None, None, None

        if train_ds is None:
            train_ds = self._build_dataset(self.splits["train"])
            val_ds = self._build_dataset(self.splits["valid"])
            test_ds = self._build_dataset(self.splits["test"])
            if cache_path:
                try:
                    torch.save({"train": train_ds, "val": val_ds, "test": test_ds}, cache_path)
                except Exception:
                    pass

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
