"""Pytest fixtures for unit and integration testing without downloading real data or training."""

import pandas as pd
import pytest


@pytest.fixture
def dummy_smiles_df() -> pd.DataFrame:
    """Toy DataFrame containing valid SMILES strings and target labels."""
    data = {
        "Drug": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
            "C1=CC=C(C=C1)CC(C(=O)O)N",  # Phenylalanine
        ],
        "Y": [1.5, 0.8, -0.2, 2.1, 0.4],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_dta_df() -> pd.DataFrame:
    """Toy DataFrame for Drug-Target Affinity."""
    data = {
        "Drug": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        ],
        "Target": [
            "MKWVTFISLLLLFSSAYSRG",
            "MAGFLKVVQLLLLSG",
            "MDAMKRGLCCVLLLCGAVFVSPS",
            "MSLFLISLLLLFSSAYSRG",
        ],
        "Y": [6.5, 7.2, 5.8, 8.1],
    }
    return pd.DataFrame(data)
