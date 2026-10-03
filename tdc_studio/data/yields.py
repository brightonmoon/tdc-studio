"""TDC Yields dataset module for reaction yield prediction (Buchwald-Hartwig, USPTO)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule

logger = logging.getLogger("tdc_studio.data.yields")


def get_mock_yields_dataset(n_samples: int = 50) -> pd.DataFrame:
    """Generate synthetic Buchwald-Hartwig coupling reactions with realistic yields."""
    reactions = [
        ("c1ccc(I)cc1.c1ccc(N)cc1", 85.4),
        ("c1ccc(Br)cc1.c1ccc(N)cc1", 72.1),
        ("c1ccc(Cl)cc1.c1ccc(N)cc1", 45.0),
        ("Cc1ccc(Br)cc1.CNc1ccccc1", 68.3),
        ("COc1ccc(Br)cc1.c1ccc(N)cc1", 79.5),
        ("c1ccc(Br)cc1.C1CCNCC1", 91.2),
        ("c1ccc(Br)cc1.C1COCCN1", 88.0),
        ("O=C(c1ccc(Br)cc1)N.c1ccc(N)cc1", 55.6),
        ("FC(F)(F)c1ccc(Br)cc1.c1ccc(N)cc1", 62.4),
        ("c1ccc(Br)cc1.NC1CCCCC1", 81.0),
    ]
    records = []
    for i in range(n_samples):
        base_rxn, base_yield = reactions[i % len(reactions)]
        # Add slight natural jitter
        jittered_yield = max(0.0, min(100.0, base_yield + (i % 7 - 3) * 1.5))
        records.append(
            {
                "reaction": base_rxn,
                "yield": jittered_yield,
            }
        )
    return pd.DataFrame(records)


class ReactionYieldDataset(Dataset):
    """PyTorch Dataset for chemical reaction yield prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        n_bits: int = 2048,
    ):
        self.df = df.reset_index(drop=True)
        self.n_bits = n_bits

    def __len__(self) -> int:
        return len(self.df)

    def _reaction_to_fingerprint(self, rxn_str: str) -> torch.Tensor:
        """Compute differential/combined Morgan fingerprint for reaction components."""
        parts = rxn_str.split(".")
        fp_accum = np.zeros(self.n_bits, dtype=np.float32)
        valid_mols = 0
        for part in parts:
            mol = Chem.MolFromSmiles(part.strip())
            if mol is not None:
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits)
                fp_accum += np.array(bv, dtype=np.float32)
                valid_mols += 1
        if valid_mols > 0:
            fp_accum /= valid_mols
        return torch.tensor(fp_accum, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rxn_str = str(row["reaction"] if "reaction" in row else row.get("input", ""))
        yield_val = float(row["yield"] if "yield" in row else row.get("Y", 0.0))

        fp = self._reaction_to_fingerprint(rxn_str)
        # Normalize yield to 0.0 ~ 1.0 range
        norm_yield = yield_val / 100.0 if yield_val > 1.0 else yield_val

        return {
            "reaction": rxn_str,
            "features": fp,
            "yield": torch.tensor(norm_yield, dtype=torch.float32),
            "yield_pct": torch.tensor(yield_val, dtype=torch.float32),
        }


def yields_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of reaction yield samples into batched tensors."""
    features = torch.stack([item["features"] for item in batch])
    yields = torch.stack([item["yield"] for item in batch])
    yields_pct = torch.stack([item["yield_pct"] for item in batch])

    return {
        "reactions": [item["reaction"] for item in batch],
        "features": features,
        "yield": yields,
        "yield_pct": yields_pct,
    }


@DATASETS.register("yields_loader")
class YieldsDataModule(BaseTDCDataModule):
    """DataModule for TDC reaction yield datasets (Buchwald-Hartwig, USPTO)."""

    def __init__(
        self,
        dataset_name: str = "Buchwald-Hartwig",
        split_type: str = "random",
        seed: int = 42,
        n_bits: int = 2048,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type="regression",
            metric_name="r2",
            synthetic_df=synthetic_df,
        )
        self.n_bits = n_bits

    def prepare_data(self) -> None:
        """Download or prepare the yield dataset splits."""
        if self.synthetic_df is not None:
            df = self.synthetic_df
        else:
            try:
                from tdc.single_pred import Yields

                logger.info(f"Loading '{self.dataset_name}' via PyTDC...")
                data = Yields(name=self.dataset_name)
                raw_splits = data.get_split(method=self.split_type, seed=self.seed)
                self.splits = {k: v.reset_index(drop=True) for k, v in raw_splits.items()}
                self.is_prepared = True
                return
            except (ImportError, Exception) as exc:
                logger.warning(
                    f"PyTDC unavailable or download failed ({exc}). Using synthetic fallback."
                )
                df = get_mock_yields_dataset(n_samples=50)

        n = len(df)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))

        shuffled = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        self.splits = {
            "train": shuffled.iloc[:n_train].reset_index(drop=True),
            "valid": shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True),
            "test": shuffled.iloc[n_train + n_val :].reset_index(drop=True),
        }
        self.is_prepared = True

    def setup_loaders(
        self, batch_size: int = 16, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for yield regression."""
        self.check_prepared()

        train_ds = ReactionYieldDataset(self.splits["train"], n_bits=self.n_bits)
        val_ds = ReactionYieldDataset(self.splits["valid"], n_bits=self.n_bits)
        test_ds = ReactionYieldDataset(self.splits["test"], n_bits=self.n_bits)

        return (
            DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=yields_collate_fn,
                num_workers=num_workers,
            ),
            DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=yields_collate_fn,
                num_workers=num_workers,
            ),
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=yields_collate_fn,
                num_workers=num_workers,
            ),
        )
