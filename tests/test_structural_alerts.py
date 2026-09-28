"""Unit tests for Ashby-Tennant 100-dimensional structural alerts module."""

import numpy as np
from rdkit import Chem

from tdc_studio.features.structural_alerts import (
    AshbyTennantAlertExtractor,
    get_ashby_tennant_extractor,
)


def test_extractor_initialization_and_dimension():
    extractor = get_ashby_tennant_extractor()
    assert extractor.dim == 100
    assert len(extractor.feature_names) == 100
    assert len(set(extractor.feature_names)) == 100  # Unique names


def test_nitro_and_halide_alerts():
    extractor = AshbyTennantAlertExtractor()
    # Nitrobenzene: c1ccccc1[N+](=O)[O-]
    nitro_vec = extractor.extract_from_smiles("c1ccccc1[N+](=O)[O-]")
    assert isinstance(nitro_vec, np.ndarray)
    assert nitro_vec.shape == (100,)
    assert nitro_vec.dtype == np.float32

    mol = Chem.MolFromSmiles("c1ccccc1[N+](=O)[O-]")
    matched = extractor.get_matched_alerts(mol)
    assert "aromatic_nitro" in matched

    # Benzyl chloride: c1ccccc1CCl
    benzyl_vec = extractor.extract_from_smiles("c1ccccc1CCl")
    assert benzyl_vec.shape == (100,)
    mol_benzyl = Chem.MolFromSmiles("c1ccccc1CCl")
    matched_benzyl = extractor.get_matched_alerts(mol_benzyl)
    assert "benzyl_halide" in matched_benzyl


def test_epoxide_and_aziridine():
    extractor = get_ashby_tennant_extractor()
    # Ethylene oxide (oxirane): C1CO1
    epoxide_vec = extractor.extract_from_smiles("C1CO1")
    assert epoxide_vec.sum() > 0
    mol = Chem.MolFromSmiles("C1CO1")
    matched = extractor.get_matched_alerts(mol)
    assert "epoxide_aliphatic" in matched


def test_empty_or_invalid_smiles():
    extractor = get_ashby_tennant_extractor()
    empty_vec = extractor.extract_from_smiles("")
    assert empty_vec.shape == (100,)
    assert empty_vec.sum() == 0.0

    invalid_vec = extractor.extract_from_smiles("INVALID_SMILES_STRING")
    assert invalid_vec.shape == (100,)
    assert invalid_vec.sum() == 0.0


def test_return_counts():
    extractor = get_ashby_tennant_extractor()
    # Dinitrobenzene: 2 aromatic nitro groups
    mol = Chem.MolFromSmiles("O=[N+]([O-])c1cccc([N+](=O)[O-])c1")
    count_vec = extractor.extract(mol, return_counts=True)
    binary_vec = extractor.extract(mol, return_counts=False)

    nitro_idx = extractor.feature_names.index("aromatic_nitro")
    assert count_vec[nitro_idx] == 2.0
    assert binary_vec[nitro_idx] == 1.0
