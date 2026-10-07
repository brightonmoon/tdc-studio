"""Unit tests for Boltzmann 10-conformer ensemble 3D feature extraction."""

import numpy as np

from tdc_studio.features.boltzmann_conformers import (
    BOLTZMANN_FEATURE_NAMES,
    batch_extract_boltzmann_features,
    compute_boltzmann_conformer_features,
    extract_boltzmann_conformer_vector,
)


def test_boltzmann_conformer_features_valid_molecule():
    """Verify Boltzmann 3D features on Aspirin and Ibuprofen."""
    # Aspirin
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    feats = compute_boltzmann_conformer_features(smi, num_confs=10, random_seed=42)

    assert isinstance(feats, dict)
    assert len(feats) == len(BOLTZMANN_FEATURE_NAMES)
    for name in BOLTZMANN_FEATURE_NAMES:
        assert name in feats
        assert isinstance(feats[name], float)
        assert np.isfinite(feats[name])

    assert feats["boltzmann_pbf"] >= 0.0
    assert 0.0 <= feats["boltzmann_spherocity"] <= 1.0
    assert feats["conformer_energy_span"] >= 0.0
    assert feats["conformational_entropy"] >= 0.0


def test_extract_boltzmann_conformer_vector():
    """Verify vector format and dimension."""
    smi = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    vec = extract_boltzmann_conformer_vector(smi, num_confs=5)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_boltzmann_conformer_graceful_fallback():
    """Verify robust fallback on invalid SMILES and single atoms."""
    # Invalid SMILES
    bad_feats = compute_boltzmann_conformer_features("INVALID_SMILES")
    for name in BOLTZMANN_FEATURE_NAMES:
        assert bad_feats[name] == 0.0

    # Single atom (no bonds for 3D PBF)
    single_atom = compute_boltzmann_conformer_features("[Na+]")
    assert single_atom["boltzmann_pbf"] == 0.0


def test_batch_extract_boltzmann_features(tmp_path):
    """Verify batch extraction and cache serialization."""
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    cache_file = tmp_path / "test_boltzmann_cache.npz"

    matrix1 = batch_extract_boltzmann_features(
        smiles_list, num_confs=5, cache_path=cache_file, verbose=False
    )
    assert matrix1.shape == (2, 8)
    assert cache_file.exists()

    # Verify reloading from cache
    matrix2 = batch_extract_boltzmann_features(
        smiles_list, num_confs=5, cache_path=cache_file, verbose=False
    )
    np.testing.assert_array_almost_equal(matrix1, matrix2)
