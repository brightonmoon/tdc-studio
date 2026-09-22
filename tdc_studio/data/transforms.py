"""Transforms for molecular graphs, biological sequences, SMILES, and fingerprints."""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data

from tdc_studio.core.registry import TRANSFORMS

# Standard atomic numbers for drug-like molecules (Legacy 10 types)
DEFAULT_ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

# Extended atomic numbers (20 common elements in therapeutics)
EXTENDED_ATOM_LIST = [1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 26, 29, 30, 35, 53]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Standard SMILES regex token pattern (Schwaller et al., Molecular Transformer)
SMILES_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


def atom_to_feature(atom: Chem.Atom, extended: bool = False) -> List[float]:
    """Convert RDKit Atom to numerical feature vector.

    If extended=False, returns standard 14-dim vector for backward compatibility.
    If extended=True, returns rich 37-dim vector (atom type, degree, hybridization, etc.).
    """
    if not extended:
        atomic_num = atom.GetAtomicNum()
        one_hot = [1.0 if atomic_num == a else 0.0 for a in DEFAULT_ATOM_LIST]
        extra = [
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
        ]
        return one_hot + extra

    # Extended 37-dim atom features
    atomic_num = atom.GetAtomicNum()
    atom_type_oh = [1.0 if atomic_num == a else 0.0 for a in EXTENDED_ATOM_LIST]
    is_unk = 1.0 if atomic_num not in EXTENDED_ATOM_LIST else 0.0
    atom_type_oh.append(is_unk)  # 21 dims

    degree = atom.GetTotalDegree()
    degree_oh = [1.0 if degree == d else 0.0 for d in range(6)]  # 6 dims (0 to 5+)

    formal_charge = float(atom.GetFormalCharge())  # 1 dim

    hyb = atom.GetHybridization()
    hyb_list = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb_oh = [1.0 if hyb == h else 0.0 for h in hyb_list]
    hyb_oh.append(1.0 if hyb not in hyb_list else 0.0)  # 6 dims

    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0  # 1 dim
    num_hs = float(atom.GetTotalNumHs())  # 1 dim
    in_ring = 1.0 if atom.IsInRing() else 0.0  # 1 dim

    return atom_type_oh + degree_oh + [formal_charge] + hyb_oh + [is_aromatic, num_hs, in_ring]


def bond_to_feature(bond: Chem.Bond) -> List[float]:
    """Convert RDKit Bond to 6-dimensional feature vector.

    [Single, Double, Triple, Aromatic, IsConjugated, IsInRing]
    """
    b_type = bond.GetBondType()
    b_type_oh = [1.0 if b_type == bt else 0.0 for bt in BOND_TYPES]
    is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
    in_ring = 1.0 if bond.IsInRing() else 0.0
    return b_type_oh + [is_conjugated, in_ring]


@TRANSFORMS.register("smiles_to_graph")
class SmilesToGraphTransform:
    """Transform SMILES string into PyTorch Geometric Data object with node and edge features."""

    def __init__(self, add_self_loops: bool = False, extended: bool = False):
        self.add_self_loops = add_self_loops
        self.extended = extended

    def __call__(self, smiles: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. Node features
        atom_feats = [atom_to_feature(atom, extended=self.extended) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)

        # 2. Edge indices and edge features
        edge_indices: List[List[int]] = [[], []]
        edge_attrs: List[List[float]] = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            b_feat = bond_to_feature(bond)

            # Bidirectional graph edges
            edge_indices[0].extend([i, j])
            edge_indices[1].extend([j, i])
            edge_attrs.extend([b_feat, b_feat])

        if len(edge_indices[0]) == 0:
            # Molecule with single atom (no bonds)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = x.size(0)
        return data


@TRANSFORMS.register("smiles_tokenizer")
class SmilesTokenizer:
    """Regex-based SMILES tokenizer using the Molecular Transformer grammar."""

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 256,
    ):
        self.regex = re.compile(SMILES_REGEX_PATTERN)
        self.max_length = max_length

        if vocab is None:
            # Standard chemical token dictionary
            special_tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask}"]
            standard_tokens = [
                "C",
                "c",
                "N",
                "n",
                "O",
                "o",
                "S",
                "s",
                "P",
                "p",
                "F",
                "Cl",
                "Br",
                "I",
                "B",
                "b",
                "H",
                "K",
                "Na",
                "Li",
                "Ca",
                "Zn",
                "Fe",
                "Si",
                "Se",
                "se",
                "Te",
                "te",
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
                "/",
                "\\",
                "@",
                "@@",
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
                "%10",
                "%11",
                "%12",
                "%13",
                "%14",
                "%15",
                "[nH]",
                "[O-]",
                "[N+]",
                "[NH+]",
                "[NH2+]",
                "[NH3+]",
                "[S-]",
                "[P+]",
                "[Si]",
                "[se]",
                "[te]",
                "[B-]",
                "[C-]",
                "[CH-]",
                "[CH2-]",
                "[OH]",
            ]
            all_tokens = special_tokens + standard_tokens
            # Dedup preserving order
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
        """Split SMILES into atom and bond tokens via regex."""
        return [token for token in self.regex.findall(smiles)]

    def encode(self, smiles: str, max_length: Optional[int] = None) -> torch.Tensor:
        """Tokenize and convert SMILES to 1D integer Tensor with <cls> and <sep>."""
        max_len = max_length or self.max_length
        tokens = self.tokenize(smiles)

        token_ids = [self.cls_token_id]
        token_ids.extend([self.vocab.get(tok, self.unk_token_id) for tok in tokens[: max_len - 2]])
        token_ids.append(self.sep_token_id)

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a SMILES string."""
        specials = {self.pad_token_id, self.cls_token_id, self.sep_token_id}
        parts = []
        for tid in token_ids:
            tid_int = int(tid)
            if skip_special_tokens and tid_int in specials:
                continue
            parts.append(self.inv_vocab.get(tid_int, "<unk>"))
        return "".join(parts)

    def __call__(self, smiles: str) -> torch.Tensor:
        return self.encode(smiles)


@TRANSFORMS.register("morgan_fingerprint")
class MorganFingerprintTransform:
    """Generate circular Morgan Fingerprints (ECFP) from SMILES."""

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, smiles: str) -> Optional[torch.Tensor]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        arr = list(fp)
        return torch.tensor(arr, dtype=torch.float32)


@TRANSFORMS.register("sequence_tokenizer")
class SequenceTokenizer:
    """Character-level tokenizer for amino acid sequences (Proteins/Targets)."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_length: int = 512):
        if vocab is None:
            # Common 20 standard amino acids + special tokens
            tokens = [
                "<pad>",
                "<unk>",
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
            ]
            self.vocab = {t: i for i, t in enumerate(tokens)}
        else:
            self.vocab = vocab
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.unk_token_id = self.vocab.get("<unk>", 1)
        self.max_length = max_length

    def __call__(self, seq: str) -> torch.Tensor:
        tokens = [self.vocab.get(ch, self.unk_token_id) for ch in seq[: self.max_length]]
        return torch.tensor(tokens, dtype=torch.long)


@TRANSFORMS.register("canonical_smiles_normalizer")
class CanonicalSmilesNormalizer:
    """Sanitize SMILES, strip counterions/salts (retain largest organic fragment), and canonicalize."""

    def __init__(self, remove_salts: bool = True, isomeric: bool = True):
        self.remove_salts = remove_salts
        self.isomeric = isomeric

    def __call__(self, smiles: str) -> Optional[str]:
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if self.remove_salts and "." in smiles:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, isomericSmiles=self.isomeric, canonical=True)


@TRANSFORMS.register("randomized_smiles_augmenter")
class RandomizedSmilesAugmenter:
    """Generate randomized SMILES for molecular sequence data augmentation."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, smiles: str) -> str:
        import random

        if random.random() > self.p:
            return smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


@TRANSFORMS.register("rdkit_2d_descriptors")
class RDKit2DDescriptorsTransform:
    """Extract 210 RDKit 2D Physico-chemical descriptors (Chemprop/DMPNN-Des baseline)."""

    def __init__(self, fill_na: float = 0.0):
        self.fill_na = fill_na

    def __call__(self, smiles: str) -> Optional[torch.Tensor]:
        if not smiles or not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            vals = []
            for v in desc_dict.values():
                if v is None or np.isnan(v) or np.isinf(v):
                    vals.append(self.fill_na)
                else:
                    vals.append(float(v))
            t = torch.tensor(vals, dtype=torch.float32)
            return torch.nan_to_num(t, nan=self.fill_na, posinf=1e5, neginf=-1e5)
        except Exception:
            return None


