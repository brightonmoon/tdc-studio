"""Single-prediction dataset loaders (ADMET, Tox, HTS)."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule, MolecularDataset
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import (
    MorganFingerprintTransform,
    SmilesToGraphTransform,
    SmilesTokenizer,
)

_TDC_DATAVERSE_MAP = {
    "caco2_wang": 4259569,
    "lipophilicity_astrazeneca": 4259575,
    "solubility_aqsoldb": 4259576,
    "hia_hou": 4259574,
    "herg": 4259581,
    "ames": 4259565,
    "bbb_martins": 4259566,
}


def _scaffold_split(
    df: pd.DataFrame, smiles_col: str = "Drug", seed: int = 42
) -> Dict[str, pd.DataFrame]:
    from collections import defaultdict

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        s = str(row[smiles_col])
        mol = Chem.MolFromSmiles(s)
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        scaffolds[scaff].append(idx)

    sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
    train_indices, val_indices, test_indices = [], [], []
    n_total = len(df)
    n_train_target = int(n_total * 0.7)
    n_val_target = int(n_total * 0.1)

    for group in sorted_scaffolds:
        if len(train_indices) + len(group) <= n_train_target:
            train_indices.extend(group)
        elif len(val_indices) + len(group) <= n_val_target:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    return {
        "train": df.loc[train_indices].reset_index(drop=True),
        "valid": df.loc[val_indices].reset_index(drop=True),
        "test": df.loc[test_indices].reset_index(drop=True),
    }


def _load_tdc_fallback(
    dataset_name: str, split_type: str = "scaffold", seed: int = 42
) -> Dict[str, pd.DataFrame]:
    import io
    from pathlib import Path

    import requests

    cache_dir = Path("./data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}.tsv"

    if cache_file.exists():
        df = pd.read_csv(cache_file, sep="\t")
    else:
        file_id = _TDC_DATAVERSE_MAP.get(dataset_name.lower())
        if not file_id:
            raise RuntimeError(f"Unknown dataset '{dataset_name}' for fallback download.")
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        df.to_csv(cache_file, sep="\t", index=False)

    smiles_col = "Drug" if "Drug" in df.columns else ("smiles" if "smiles" in df.columns else df.columns[1])
    if split_type == "scaffold":
        return _scaffold_split(df, smiles_col=smiles_col, seed=seed)
    else:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)
        return {
            "train": shuffled.iloc[:n_train],
            "valid": shuffled.iloc[n_train : n_train + n_val],
            "test": shuffled.iloc[n_train + n_val :],
        }


@DATASETS.register("admet_loader")
class ADMETDataModule(BaseTDCDataModule):
    """ADMET dataset module supporting Graph, Sequence, and Fingerprint modalities."""

    def __init__(
        self,
        dataset_name: str = "caco2_wang",
        split_type: str = "scaffold",
        seed: int = 42,
        task_type: str = "regression",
        metric_name: str = "mae",
        modality: str = "graph",
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
        self.modality = modality.lower()
        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()

    def prepare_data(self) -> None:
        """Load dataset from synthetic data or TDC library with Dataverse fallback."""
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
            except Exception:
                # Direct Dataverse download & scaffold split fallback
                self.splits = _load_tdc_fallback(
                    dataset_name=self.dataset_name,
                    split_type=self.split_type,
                    seed=self.seed,
                )
        self.is_prepared = True

    def _build_dataset(self, df: pd.DataFrame) -> MolecularDataset:
        samples: List[Dict[str, Any]] = []
        smiles_col = "Drug" if "Drug" in df.columns else "smiles"
        label_col = "Y" if "Y" in df.columns else "label"

        for _, row in df.iterrows():
            s = str(row[smiles_col])
            lbl = float(row[label_col]) if label_col in row else 0.0

            if self.modality == "graph":
                g = self.graph_transform(s)
                if g is not None:
                    samples.append({"drug_graph": g, "label": lbl})
            elif self.modality == "sequence":
                seq = self.smiles_tokenizer(s)
                samples.append({"smiles_seq": seq, "label": lbl})
            elif self.modality == "fingerprint":
                fp = self.fingerprint_transform(s)
                if fp is not None:
                    samples.append({"fingerprint": fp, "label": lbl})
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


@DATASETS.register("tox_loader")
class ToxDataModule(BaseTDCDataModule):
    """Toxicology dataset module (Tox) supporting Graph, Sequence, and Fingerprint modalities."""

    def __init__(
        self,
        dataset_name: str = "herg",
        split_type: str = "scaffold",
        seed: int = 42,
        task_type: str = "binary_classification",
        metric_name: str = "roc_auc",
        modality: str = "graph",
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
        self.modality = modality.lower()
        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()

    def prepare_data(self) -> None:
        """Load toxicology dataset from synthetic data or TDC Tox package."""
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
                from tdc.single_pred import Tox

                data = Tox(name=self.dataset_name)
                self.splits = data.get_split(method=self.split_type, seed=self.seed)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load TDC Tox dataset '{self.dataset_name}'. "
                    f"Ensure PyTDC is installed and network is available, or provide synthetic_df. Error: {e}"
                )
        self.is_prepared = True

    def _build_dataset(self, df: pd.DataFrame) -> MolecularDataset:
        samples: List[Dict[str, Any]] = []
        smiles_col = "Drug" if "Drug" in df.columns else "smiles"
        label_col = "Y" if "Y" in df.columns else "label"

        for _, row in df.iterrows():
            s = str(row[smiles_col])
            lbl = float(row[label_col]) if label_col in row else 0.0

            if self.modality == "graph":
                g = self.graph_transform(s)
                if g is not None:
                    samples.append({"drug_graph": g, "label": lbl})
            elif self.modality == "sequence":
                seq = self.smiles_tokenizer(s)
                samples.append({"smiles_seq": seq, "label": lbl})
            elif self.modality == "fingerprint":
                fp = self.fingerprint_transform(s)
                if fp is not None:
                    samples.append({"fingerprint": fp, "label": lbl})
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
