"""TDC RetroSyn dataset module and reaction sequence processing."""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from tdc_studio.core.registry import DATASETS, TRANSFORMS
from tdc_studio.data.base import BaseTDCDataModule

logger = logging.getLogger("tdc_studio.data.retrosyn")

# Regex pattern for SMILES and Reaction tokenization (based on Molecular Transformer)
REACTION_TOKEN_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

# Standard USPTO-50K 10 Reaction Classes
USPTO_50K_REACTION_CLASSES = {
    1: "Heteroatom alkylation and arylation",
    2: "Acylation and related processes",
    3: "C-C bond formation",
    4: "Heterocycle formation",
    5: "Protections",
    6: "Deprotections",
    7: "Reductions",
    8: "Oxidations",
    9: "Functional group interconversion (FGI)",
    10: "Functional group addition (FGA)",
}


def remove_atom_mapping(smiles: str) -> str:
    """Remove atom mapping numbers (e.g., [CH3:1] -> C, [N:2] -> N) and canonicalize."""
    if not smiles or not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback to regex stripping of mapping numbers if initial parse fails
        clean_smiles = re.sub(r":\d+\]", "]", smiles)
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return clean_smiles
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_reaction_smiles(product_smiles: str, reactants_smiles: str) -> Tuple[str, str]:
    """Canonicalize both product and reactants, sorting reactant fragments for invariance."""
    clean_prod = remove_atom_mapping(product_smiles)

    # Process reactant fragments separated by '.'
    reactant_frags = reactants_smiles.split(".")
    cleaned_frags = []
    for frag in reactant_frags:
        clean_frag = remove_atom_mapping(frag.strip())
        if clean_frag:
            cleaned_frags.append(clean_frag)
    # Sort reactant fragments lexicographically for deterministic order
    sorted_reactants = ".".join(sorted(cleaned_frags))
    return clean_prod, sorted_reactants


@TRANSFORMS.register("reaction_tokenizer")
class ReactionTokenizer:
    """Sequence tokenizer specialized for chemical reactions and retrosynthesis."""

    DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<rxn>"]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 256,
    ):
        self.regex = REACTION_TOKEN_REGEX
        self.max_length = max_length

        if vocab is None:
            # Build comprehensive chemical token vocabulary
            special_tokens = list(self.DEFAULT_SPECIAL_TOKENS)
            # Add reaction class tokens <RX_1> to <RX_10>
            reaction_class_tokens = [f"<RX_{i}>" for i in range(1, 11)]

            chem_tokens = [
                "C",
                "N",
                "O",
                "S",
                "P",
                "F",
                "Cl",
                "Br",
                "I",
                "B",
                "c",
                "n",
                "o",
                "s",
                "p",
                "(",
                ")",
                "[",
                "]",
                "=",
                "#",
                "-",
                "+",
                ":",
                ".",
                "@",
                "@@",
                "/",
                "\\",
                "%",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "[nH]",
                "[O-]",
                "[N+]",
                "[NH+]",
                "[NH2+]",
                "[NH3+]",
                "[S-]",
                "[P+]",
                "[B-]",
                "[F-]",
                "[Cl-]",
                "[Br-]",
                "[I-]",
                "[Na+]",
                "[K+]",
                "[Li+]",
                "[Mg+2]",
                "[Zn+2]",
                "[Si]",
                "[se]",
                "[te]",
                "[H]",
                "[2H]",
            ]
            all_tokens = special_tokens + reaction_class_tokens + chem_tokens
            unique_tokens = list(dict.fromkeys(all_tokens))
            self.vocab = {t: i for i, t in enumerate(unique_tokens)}
        else:
            self.vocab = vocab

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.unk_token_id = self.vocab.get("<unk>", 1)
        self.cls_token_id = self.vocab.get("<cls>", 2)
        self.sep_token_id = self.vocab.get("<sep>", 3)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, smiles: str) -> List[str]:
        """Split reaction or molecule SMILES into chemical tokens."""
        return [token for token in self.regex.findall(smiles)]

    def encode(
        self,
        smiles: str,
        reaction_type: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize SMILES into 1D integer Tensor with attention mask.

        If reaction_type is provided (1~10), prefixes with class token <RX_{type}>.
        Returns:
            token_ids: LongTensor of shape [max_length]
            attention_mask: LongTensor of shape [max_length] (1 for valid, 0 for pad)
        """
        max_len = max_length or self.max_length
        tokens = self.tokenize(smiles)

        token_list = [self.cls_token_id]
        if reaction_type is not None and 1 <= reaction_type <= 10:
            rx_token = f"<RX_{reaction_type}>"
            token_list.append(self.vocab.get(rx_token, self.unk_token_id))

        for tok in tokens[: max_len - 2]:
            if len(token_list) >= max_len - 1:
                break
            token_list.append(self.vocab.get(tok, self.unk_token_id))
        token_list.append(self.sep_token_id)

        seq_len = len(token_list)
        pad_len = max_len - seq_len
        if pad_len > 0:
            token_ids = token_list + [self.pad_token_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
        else:
            token_ids = token_list[:max_len]
            attention_mask = [1] * max_len

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a SMILES string."""
        specials = {
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.vocab.get("<mask>", 4),
        }
        parts = []
        for tid in token_ids:
            tid_int = int(tid)
            tok = self.inv_vocab.get(tid_int, "")
            if skip_special_tokens and (tid_int in specials or tok.startswith("<RX_")):
                continue
            parts.append(tok)
        return "".join(parts)


def get_mock_retrosyn_dataset(n_samples: int = 60) -> pd.DataFrame:
    """Generate verified organic reaction samples covering 10 USPTO-50K reaction classes."""
    verified_reactions = [
        # 1. Alkylation/Arylation: Benzyl bromide + Aniline -> N-benzylaniline
        ("c1ccc(CNCc2ccccc2)cc1", "c1ccc(CBr)cc1.c1ccc(N)cc1", 1),
        ("CCOc1ccc(CC(=O)O)cc1", "CCOc1ccc(CC(=O)Cl)cc1.O", 1),
        ("Cc1ccc(NCc2ccccc2)cc1", "Cc1ccc(N)cc1.c1ccc(CBr)cc1", 1),
        ("c1ccc(OCCCC)cc1", "c1ccc(O)cc1.CCCCBr", 1),
        ("c1ccc(N(C)C)cc1", "c1ccc(N)cc1.CI", 1),
        ("c1ccc(SC)cc1", "c1ccc(S)cc1.CI", 1),
        # 2. Acylation (Amide formation): Benzoic acid + Methylamine -> N-methylbenzamide
        ("c1ccc(C(=O)NC)cc1", "c1ccc(C(=O)O)cc1.CNC", 2),
        ("CC(=O)Nc1ccccc1", "CC(=O)Cl.c1ccc(N)cc1", 2),
        ("c1ccc(C(=O)NCCc2ccccc2)cc1", "c1ccc(C(=O)O)cc1.NCc1ccccc1", 2),
        ("CC(=O)N1CCCC1", "CC(=O)Cl.C1CCCN1", 2),
        ("c1ccc(C(=O)N2CCOCC2)cc1", "c1ccc(C(=O)Cl)cc1.C1COCCN1", 2),
        ("CCOC(=O)c1ccccc1", "c1ccc(C(=O)Cl)cc1.CCO", 2),
        # 3. C-C bond formation: Suzuki coupling (Bromobenzene + Phenylboronic acid -> Biphenyl)
        ("c1ccc(-c2ccccc2)cc1", "c1ccc(Br)cc1.OB(O)c1ccccc1", 3),
        ("Cc1ccc(-c2ccccc2)cc1", "Cc1ccc(Br)cc1.OB(O)c1ccccc1", 3),
        ("COc1ccc(-c2ccccc2)cc1", "COc1ccc(Br)cc1.OB(O)c1ccccc1", 3),
        ("c1ccc(-c2ccncc2)cc1", "c1ccc(Br)cc1.OB(O)c1ccncc1", 3),
        ("c1ccc(Cc2ccccc2)cc1", "c1ccc(CBr)cc1.OB(O)c1ccccc1", 3),
        ("c1ccc(C#Cc2ccccc2)cc1", "c1ccc(I)cc1.C#Cc1ccccc1", 3),
        # 4. Heterocycle formation
        ("c1ccc2[nH]cnc2c1", "Nc1ccccc1N.O=CO", 4),
        ("c1ccc2nc[nH]c2c1", "Nc1ccccc1N.O=CO", 4),
        ("c1ccc(c2noc(-c3ccccc3)n2)cc1", "c1ccc(C(=N)NO)cc1.c1ccc(C(=O)Cl)cc1", 4),
        ("c1ccc2ocnc2c1", "Nc1ccccc1O.O=CO", 4),
        ("c1ccc2scnc2c1", "Nc1ccccc1S.O=CO", 4),
        ("c1ccc(c2nc(-c3ccccc3)cs2)cc1", "c1ccc(C(=S)N)cc1.c1ccc(C(=O)CBr)cc1", 4),
        # 5. Protection (Boc protection, etc.)
        ("CC(C)(C)OC(=O)Nc1ccccc1", "c1ccc(N)cc1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", 5),
        ("CC(C)(C)OC(=O)N1CCCC1", "C1CCCN1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", 5),
        ("CC(C)(C)[Si](C)(C)Oc1ccccc1", "c1ccc(O)cc1.CC(C)(C)[Si](C)(C)Cl", 5),
        ("c1ccc(COC(=O)Nc2ccccc2)cc1", "c1ccc(N)cc1.c1ccc(COC(=O)Cl)cc1", 5),
        ("CC(=O)Oc1ccccc1", "c1ccc(O)cc1.CC(=O)Cl", 5),
        ("CC(=O)N(C)c1ccccc1", "CNc1ccccc1.CC(=O)Cl", 5),
        # 6. Deprotection
        ("c1ccc(N)cc1", "CC(C)(C)OC(=O)Nc1ccccc1", 6),
        ("C1CCCN1", "CC(C)(C)OC(=O)N1CCCC1", 6),
        ("c1ccc(O)cc1", "CC(C)(C)[Si](C)(C)Oc1ccccc1", 6),
        ("c1ccc(C(=O)O)cc1", "CCOC(=O)c1ccccc1", 6),
        ("Cc1ccc(O)cc1", "Cc1ccc(OC(=O)C)cc1", 6),
        ("CNc1ccccc1", "CC(=O)N(C)c1ccccc1", 6),
        # 7. Reductions (Nitro to Amine, Carbonyl to Alcohol)
        ("c1ccc(N)cc1", "c1ccc([N+](=O)[O-])cc1", 7),
        ("c1ccc(CO)cc1", "c1ccc(C=O)cc1", 7),
        ("CC(O)c1ccccc1", "CC(=O)c1ccccc1", 7),
        ("Cc1ccc(N)cc1", "Cc1ccc([N+](=O)[O-])cc1", 7),
        ("c1ccc(CCN)cc1", "c1ccc(CC#N)cc1", 7),
        ("c1ccc(CCO)cc1", "c1ccc(CC(=O)O)cc1", 7),
        # 8. Oxidations (Alcohol to Aldehyde/Acid)
        ("c1ccc(C=O)cc1", "c1ccc(CO)cc1", 8),
        ("c1ccc(C(=O)O)cc1", "c1ccc(C=O)cc1", 8),
        ("CC(=O)c1ccccc1", "CC(O)c1ccccc1", 8),
        ("c1ccc(S(=O)C)cc1", "c1ccc(SC)cc1", 8),
        ("c1ccc(S(=O)(=O)C)cc1", "c1ccc(S(=O)C)cc1", 8),
        ("O=C(O)c1ccncc1", "OCc1ccncc1", 8),
        # 9. Functional group interconversion (FGI)
        ("c1ccc(Cl)cc1", "c1ccc(N)cc1", 9),
        ("c1ccc(Br)cc1", "c1ccc(N)cc1", 9),
        ("c1ccc(C#N)cc1", "c1ccc(Br)cc1", 9),
        ("c1ccc(CCl)cc1", "c1ccc(CO)cc1", 9),
        ("c1ccc(CBr)cc1", "c1ccc(CO)cc1", 9),
        ("c1ccc(CC#N)cc1", "c1ccc(CBr)cc1", 9),
        # 10. Functional group addition (FGA: Nitration, Halogenation)
        ("c1ccc([N+](=O)[O-])cc1", "c1ccccc1", 10),
        ("c1ccc(Br)cc1", "c1ccccc1", 10),
        ("c1ccc(Cl)cc1", "c1ccccc1", 10),
        ("Cc1ccc([N+](=O)[O-])cc1", "Cc1ccccc1", 10),
        ("Cc1ccc(Br)cc1", "Cc1ccccc1", 10),
        ("c1ccc(S(=O)(=O)Cl)cc1", "c1ccccc1", 10),
    ]

    records = []
    for i, (prod, react, rxn_type) in enumerate(verified_reactions[:n_samples]):
        clean_prod, clean_react = canonicalize_reaction_smiles(prod, react)
        records.append(
            {
                "input": clean_prod,
                "output": clean_react,
                "reaction_type": rxn_type,
                "reaction_name": USPTO_50K_REACTION_CLASSES.get(rxn_type, "Unknown"),
            }
        )
    return pd.DataFrame(records)


class RetroSynDataset(Dataset):
    """PyTorch Dataset for single-step retrosynthesis."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: ReactionTokenizer,
        max_length: int = 256,
        include_reaction_type: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reaction_type = include_reaction_type

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        prod_smiles = str(row["input"])
        react_smiles = str(row["output"])
        rxn_type = int(row.get("reaction_type", 0)) if self.include_reaction_type else 0

        # Encode product (encoder input)
        input_ids, attention_mask = self.tokenizer.encode(
            prod_smiles,
            reaction_type=rxn_type if self.include_reaction_type else None,
            max_length=self.max_length,
        )

        # Encode target reactants (decoder output)
        target_ids, target_mask = self.tokenizer.encode(
            react_smiles,
            max_length=self.max_length,
        )

        return {
            "input_smiles": prod_smiles,
            "output_smiles": react_smiles,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "reaction_type": torch.tensor(rxn_type, dtype=torch.long),
        }


def retrosyn_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of retrosynthesis samples into batched tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    target_mask = torch.stack([item["target_mask"] for item in batch])
    reaction_types = torch.stack([item["reaction_type"] for item in batch])

    return {
        "input_smiles": [item["input_smiles"] for item in batch],
        "output_smiles": [item["output_smiles"] for item in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "reaction_type": reaction_types,
    }


@DATASETS.register("retrosyn_loader")
class RetroSynDataModule(BaseTDCDataModule):
    """DataModule for TDC Retrosynthesis datasets (USPTO-50K, USPTO)."""

    def __init__(
        self,
        dataset_name: str = "USPTO-50K",
        split_type: str = "random",
        seed: int = 42,
        include_reaction_type: bool = True,
        max_length: int = 256,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            split_type=split_type,
            seed=seed,
            task_type="retrosynthesis",
            metric_name="top_k_exact_match",
            synthetic_df=synthetic_df,
        )
        self.include_reaction_type = include_reaction_type
        self.max_length = max_length
        self.tokenizer = ReactionTokenizer(max_length=max_length)

    def prepare_data(self) -> None:
        """Download or prepare the dataset splits."""
        if self.synthetic_df is not None:
            df = self.synthetic_df
        else:
            try:
                from tdc.generation import RetroSyn

                logger.info(f"Loading '{self.dataset_name}' via PyTDC...")
                data = RetroSyn(name=self.dataset_name)
                # TDC get_split returns dict with 'train', 'valid', 'test'
                raw_splits = data.get_split(
                    method=self.split_type,
                    seed=self.seed,
                    include_reaction_type=self.include_reaction_type,
                )
                self.splits = {k: v.reset_index(drop=True) for k, v in raw_splits.items()}
                self.is_prepared = True
                return
            except (ImportError, Exception) as exc:
                logger.warning(
                    f"PyTDC unavailable or download failed ({exc}). Using synthetic fallback."
                )
                df = get_mock_retrosyn_dataset(n_samples=60)

        # Partition dataframe into train, valid, test
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
        """Create PyTorch DataLoaders for train, valid, and test sets."""
        self.check_prepared()

        train_ds = RetroSynDataset(
            self.splits["train"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_reaction_type=self.include_reaction_type,
        )
        val_ds = RetroSynDataset(
            self.splits["valid"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_reaction_type=self.include_reaction_type,
        )
        test_ds = RetroSynDataset(
            self.splits["test"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_reaction_type=self.include_reaction_type,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=retrosyn_collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=retrosyn_collate_fn,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=retrosyn_collate_fn,
            num_workers=num_workers,
        )

        return train_loader, val_loader, test_loader
