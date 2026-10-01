"""TDC Forward Reaction dataset module (Reaction USPTO)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tdc_studio.core.registry import DATASETS
from tdc_studio.data.base import BaseTDCDataModule
from tdc_studio.data.retrosyn import (
    ReactionTokenizer,
    get_mock_retrosyn_dataset,
)

logger = logging.getLogger("tdc_studio.data.reaction")


class ForwardReactionDataset(Dataset):
    """PyTorch Dataset for forward reaction prediction (Reactants -> Product)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: ReactionTokenizer,
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        # In forward reaction, input is reactants, output is product
        reactants_smiles = str(row["output"] if "output" in row else row["input"])
        product_smiles = str(row["input"] if "output" in row else row["target"])

        input_ids, attention_mask = self.tokenizer.encode(
            reactants_smiles,
            max_length=self.max_length,
        )
        target_ids, target_mask = self.tokenizer.encode(
            product_smiles,
            max_length=self.max_length,
        )

        return {
            "reactants_smiles": reactants_smiles,
            "product_smiles": product_smiles,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }


def reaction_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of forward reaction samples into batched tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    target_mask = torch.stack([item["target_mask"] for item in batch])

    return {
        "reactants_smiles": [item["reactants_smiles"] for item in batch],
        "product_smiles": [item["product_smiles"] for item in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
    }


@DATASETS.register("reaction_loader")
class ForwardReactionDataModule(BaseTDCDataModule):
    """DataModule for TDC Forward Reaction prediction (USPTO)."""

    def __init__(
        self,
        dataset_name: str = "USPTO",
        split_type: str = "random",
        seed: int = 42,
        max_length: int = 256,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type="reaction_prediction",
            metric_name="top_k_exact_match",
            synthetic_df=synthetic_df,
        )
        self.max_length = max_length
        self.tokenizer = ReactionTokenizer(max_length=max_length)

    def prepare_data(self) -> None:
        """Download or prepare the dataset splits."""
        if self.synthetic_df is not None:
            df = self.synthetic_df
        else:
            try:
                from tdc.generation import Reaction

                logger.info(f"Loading '{self.dataset_name}' via PyTDC...")
                data = Reaction(name=self.dataset_name)
                raw_splits = data.get_split(method=self.split_type, seed=self.seed)
                self.splits = {
                    k: v.reset_index(drop=True) for k, v in raw_splits.items()
                }
                self.is_prepared = True
                return
            except (ImportError, Exception) as exc:
                logger.warning(
                    f"PyTDC unavailable or download failed ({exc}). Using synthetic fallback."
                )
                df = get_mock_retrosyn_dataset(n_samples=60)

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
        """Create PyTorch DataLoaders for forward reaction prediction."""
        self.check_prepared()

        train_ds = ForwardReactionDataset(
            self.splits["train"], tokenizer=self.tokenizer, max_length=self.max_length
        )
        val_ds = ForwardReactionDataset(
            self.splits["valid"], tokenizer=self.tokenizer, max_length=self.max_length
        )
        test_ds = ForwardReactionDataset(
            self.splits["test"], tokenizer=self.tokenizer, max_length=self.max_length
        )

        return (
            DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=reaction_collate_fn,
                num_workers=num_workers,
            ),
            DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=reaction_collate_fn,
                num_workers=num_workers,
            ),
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=reaction_collate_fn,
                num_workers=num_workers,
            ),
        )
