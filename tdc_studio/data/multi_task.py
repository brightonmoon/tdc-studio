"""Multi-task dataset loader for comprehensive ADMET and Toxicology profiles."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule, MolecularDataset
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import (
    MorganFingerprintTransform,
    SmilesToGraphTransform,
    SmilesTokenizer,
)
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss

# Canonical metadata for common TDC ADMET/Tox endpoints
KNOWN_TDC_TASKS: Dict[str, Dict[str, str]] = {
    # Absorption
    "caco2_wang": {"category": "absorption", "type": "regression", "metric": "mae"},
    "hia_hou": {"category": "absorption", "type": "binary_classification", "metric": "roc_auc"},
    "pgp_broccatelli": {"category": "absorption", "type": "binary_classification", "metric": "roc_auc"},
    "bioavailability_ma": {"category": "absorption", "type": "binary_classification", "metric": "roc_auc"},
    # Distribution
    "bbb_martins": {"category": "distribution", "type": "binary_classification", "metric": "roc_auc"},
    "ppbr_az": {"category": "distribution", "type": "regression", "metric": "mae"},
    "vdss_lombardo": {"category": "distribution", "type": "regression", "metric": "mae"},
    # Metabolism
    "cyp2d6_veith": {"category": "metabolism", "type": "binary_classification", "metric": "pr_auc"},
    "cyp3a4_veith": {"category": "metabolism", "type": "binary_classification", "metric": "pr_auc"},
    "cyp2c9_veith": {"category": "metabolism", "type": "binary_classification", "metric": "pr_auc"},
    "cyp2c19_veith": {"category": "metabolism", "type": "binary_classification", "metric": "pr_auc"},
    "cyp1a2_veith": {"category": "metabolism", "type": "binary_classification", "metric": "pr_auc"},
    # Excretion
    "half_life_obach": {"category": "excretion", "type": "regression", "metric": "spearman"},
    "clearance_hepatocyte_az": {"category": "excretion", "type": "regression", "metric": "spearman"},
    "clearance_microsome_az": {"category": "excretion", "type": "regression", "metric": "spearman"},
    # Toxicity
    "herg": {"category": "toxicity", "type": "binary_classification", "metric": "roc_auc"},
    "ames": {"category": "toxicity", "type": "binary_classification", "metric": "roc_auc"},
    "dili": {"category": "toxicity", "type": "binary_classification", "metric": "roc_auc"},
    "clintox": {"category": "toxicity", "type": "binary_classification", "metric": "roc_auc"},
    "skin_reaction": {"category": "toxicity", "type": "binary_classification", "metric": "roc_auc"},
    "carcinogens_lagunin": {"category": "toxicity", "type": "binary_classification", "metric": "roc_auc"},
    # Physicochemical
    "lipophilicity_astrazeneca": {"category": "physicochemical", "type": "regression", "metric": "mae"},
    "solubility_aqsoldb": {"category": "physicochemical", "type": "regression", "metric": "mae"},
    "hydration_free_energy": {"category": "physicochemical", "type": "regression", "metric": "mae"},
}

DEFAULT_MULTI_TASKS: List[Dict[str, str]] = [
    {"name": "caco2_wang", "category": "absorption", "type": "regression"},
    {"name": "hia_hou", "category": "absorption", "type": "binary_classification"},
    {"name": "bbb_martins", "category": "distribution", "type": "binary_classification"},
    {"name": "ppbr_az", "category": "distribution", "type": "regression"},
    {"name": "cyp3a4_veith", "category": "metabolism", "type": "binary_classification"},
    {"name": "half_life_obach", "category": "excretion", "type": "regression"},
    {"name": "herg", "category": "toxicity", "type": "binary_classification"},
    {"name": "ames", "category": "toxicity", "type": "binary_classification"},
    {"name": "dili", "category": "toxicity", "type": "binary_classification"},
    {"name": "solubility_aqsoldb", "category": "physicochemical", "type": "regression"},
]


@DATASETS.register("multitask_loader")
class MultiTaskDataModule(BaseTDCDataModule):
    """Multi-task dataset module supporting missing value masking and multi-category endpoints."""

    def __init__(
        self,
        tasks: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        dataset_name: str = "admet_multitask",
        split_type: str = "scaffold",
        seed: int = 42,
        modality: str = "graph",
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
        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()

        # Parse tasks
        self.task_configs: List[Dict[str, str]] = []
        raw_tasks = tasks if tasks is not None else DEFAULT_MULTI_TASKS

        for t in raw_tasks:
            if isinstance(t, str):
                meta = KNOWN_TDC_TASKS.get(
                    t.lower(),
                    {"category": "other", "type": "regression", "metric": "mae"},
                )
                self.task_configs.append(
                    {
                        "name": t,
                        "category": meta.get("category", "other"),
                        "type": meta.get("type", "regression"),
                    }
                )
            elif isinstance(t, dict):
                t_name = t["name"]
                meta = KNOWN_TDC_TASKS.get(t_name.lower(), {})
                self.task_configs.append(
                    {
                        "name": t_name,
                        "category": t.get("category", meta.get("category", "other")),
                        "type": t.get("type", meta.get("type", "regression")),
                    }
                )

        self.task_names = [t["name"] for t in self.task_configs]
        self.task_types = [t["type"] for t in self.task_configs]
        self.task_categories = [t["category"] for t in self.task_configs]
        self.num_tasks = len(self.task_configs)

        # Mapping category -> task indices and task names
        self.category_to_task_indices: Dict[str, List[int]] = {}
        self.category_to_task_names: Dict[str, List[str]] = {}
        for idx, t in enumerate(self.task_configs):
            cat = t["category"]
            self.category_to_task_indices.setdefault(cat, []).append(idx)
            self.category_to_task_names.setdefault(cat, []).append(t["name"])

    def get_loss_fn(self, use_uncertainty: bool = True) -> MaskedMultiTaskLoss:
        """Construct MaskedMultiTaskLoss matching this module's tasks and types."""
        return MaskedMultiTaskLoss(
            task_names=self.task_names,
            task_types=self.task_types,
            use_uncertainty=use_uncertainty,
        )

    def prepare_data(self) -> None:
        """Prepare train/valid/test splits from synthetic_df or merge TDC datasets."""
        if self.synthetic_df is not None:
            df = self.synthetic_df
            n = len(df)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            self.splits = {
                "train": df.iloc[:n_train].reset_index(drop=True),
                "valid": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
                "test": df.iloc[n_train + n_val :].reset_index(drop=True),
            }
        else:
            # Multi-dataset loader from TDC
            merged_df: Optional[pd.DataFrame] = None
            try:
                from tdc.single_pred import ADME, Tox
            except ImportError:
                raise RuntimeError(
                    "PyTDC is required for multi-task data loading when synthetic_df is not provided."
                )

            for t_name in self.task_names:
                try:
                    loader_cls = Tox if t_name in ("herg", "ames", "dili", "clintox", "skin_reaction") else ADME
                    data = loader_cls(name=t_name)
                    df_t = data.get_data()
                    sub_df = df_t[["Drug", "Y"]].rename(columns={"Y": t_name})
                    if merged_df is None:
                        merged_df = sub_df
                    else:
                        merged_df = pd.merge(merged_df, sub_df, on="Drug", how="outer")
                except Exception as e:
                    raise RuntimeError(f"Failed to fetch TDC dataset '{t_name}': {e}")

            if merged_df is None or len(merged_df) == 0:
                raise RuntimeError("No data could be extracted for the specified multi-task set.")

            # Simple scaffold or random split
            merged_df = merged_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
            n = len(merged_df)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            self.splits = {
                "train": merged_df.iloc[:n_train].reset_index(drop=True),
                "valid": merged_df.iloc[n_train : n_train + n_val].reset_index(drop=True),
                "test": merged_df.iloc[n_train + n_val :].reset_index(drop=True),
            }

        self.is_prepared = True

    def _build_dataset(self, df: pd.DataFrame) -> MolecularDataset:
        samples: List[Dict[str, Any]] = []
        smiles_col = "Drug" if "Drug" in df.columns else "smiles"

        for _, row in df.iterrows():
            s = str(row[smiles_col])
            labels: List[float] = []
            mask: List[bool] = []

            for t_name in self.task_names:
                if t_name in row and pd.notna(row[t_name]):
                    labels.append(float(row[t_name]))
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
            else:
                raise ValueError(
                    f"Unsupported modality '{self.modality}'. Choose from 'graph', 'sequence', 'fingerprint'."
                )

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
