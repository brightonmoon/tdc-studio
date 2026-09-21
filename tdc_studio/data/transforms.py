"""Transforms for molecular graphs and biological sequences."""

from typing import Dict, List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

from tdc_studio.core.registry import TRANSFORMS

# Standard atomic numbers for drug-like molecules
DEFAULT_ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


def atom_to_feature(atom: Chem.Atom) -> List[float]:
    """Convert RDKit Atom to basic numerical feature vector."""
    atomic_num = atom.GetAtomicNum()
    one_hot = [1.0 if atomic_num == a else 0.0 for a in DEFAULT_ATOM_LIST]
    # Other features: degree, formal charge, hybridization, aromaticity
    extra = [
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]
    return one_hot + extra


@TRANSFORMS.register("smiles_to_graph")
class SmilesToGraphTransform:
    """Transform SMILES string into PyTorch Geometric Data object."""

    def __init__(self, add_self_loops: bool = False):
        self.add_self_loops = add_self_loops

    def __call__(self, smiles: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features
        atom_feats = [atom_to_feature(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)

        # Edge indices
        edge_indices: List[List[int]] = [[], []]
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices[0].extend([i, j])
            edge_indices[1].extend([j, i])

        if len(edge_indices[0]) == 0:
            # Molecule with single atom
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = x.size(0)
        return data


@TRANSFORMS.register("sequence_tokenizer")
class SequenceTokenizer:
    """Character-level tokenizer for amino acid sequences or SMILES."""

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
