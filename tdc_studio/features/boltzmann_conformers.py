"""Boltzmann 10-Conformer Ensemble 3D Feature Extraction Module.

Generates an ensemble of conformers (default 10) for each molecule,
optimizes geometries via MMFF94 force field, computes Boltzmann weights
from relative conformational energies, and derives Boltzmann-weighted average
3D steric descriptors (PBF, Spherocity, Asphericity, etc.) as well as
conformational flexibility/entropy metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Boltzmann constant * 298.15 K in kcal/mol: kB * T = 1.987204e-3 * 298.15 = 0.5924855 kcal/mol
KB_T_298K = 0.5924855

FEATURE_NAMES = [
    "boltzmann_pbf",
    "boltzmann_spherocity",
    "boltzmann_asphericity",
    "boltzmann_eccentricity",
    "boltzmann_inertial_shape_factor",
    "boltzmann_radius_of_gyration",
    "conformer_energy_span",
    "conformational_entropy",
]

BOLTZMANN_FEATURE_NAMES = FEATURE_NAMES


def compute_boltzmann_conformer_features(
    mol_or_smiles: Union[Chem.Mol, str],
    num_confs: int = 10,
    temperature: float = 298.15,
    max_attempts: int = 50,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Extract Boltzmann-weighted 3D conformer ensemble features for a molecule.

    Args:
        mol_or_smiles: RDKit Mol object or SMILES string.
        num_confs: Target number of conformers in the ensemble (default: 10).
        temperature: Kelvin temperature for Boltzmann weighting (default: 298.15 K).
        max_attempts: Maximum attempts for ETKDG conformer generation.
        random_seed: Seed for stochastic conformer embedding.

    Returns:
        Dictionary mapping descriptor names to their Boltzmann-weighted float values.
    """
    default_vals = {name: 0.0 for name in FEATURE_NAMES}

    if isinstance(mol_or_smiles, str):
        if not mol_or_smiles:
            return default_vals
        try:
            mol = Chem.MolFromSmiles(mol_or_smiles)
        except Exception:
            return default_vals
    else:
        mol = mol_or_smiles

    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return default_vals

    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.maxAttempts = max_attempts
        params.pruneRmsThresh = 0.35  # Filter near-duplicate conformers

        cids = list(AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params))

        # Fallback to single conformer if multiple embedding fails
        if len(cids) == 0:
            params.pruneRmsThresh = -1.0
            single_cid = AllChem.EmbedMolecule(mol_3d, params)
            if single_cid >= 0:
                cids = [single_cid]

        if len(cids) == 0:
            return default_vals

        # MMFF94 optimization & energy extraction
        mp = AllChem.MMFFGetMoleculeProperties(mol_3d, mmffVariant="MMFF94")
        energies = []
        valid_cids = []

        for cid in cids:
            try:
                # Optimize conformer
                AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid, maxIters=100)
                if mp is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol_3d, mp, confId=cid)
                    if ff is not None:
                        e = ff.CalcEnergy()
                        if np.isfinite(e):
                            energies.append(e)
                            valid_cids.append(cid)
                            continue
                # If force field energy fails, treat as neutral energy
                energies.append(0.0)
                valid_cids.append(cid)
            except Exception:
                pass

        if len(valid_cids) == 0:
            return default_vals

        energies = np.array(energies, dtype=np.float64)
        min_e = np.min(energies)
        delta_e = energies - min_e  # Relative energy in kcal/mol

        # Thermal energy kB * T
        rt = (1.987204e-3) * temperature
        # Numerical stability clamp: prevent overflow in exp
        clipped_exponent = np.clip(-delta_e / rt, -50.0, 0.0)
        unnorm_weights = np.exp(clipped_exponent)
        sum_weights = np.sum(unnorm_weights)
        if sum_weights > 0:
            weights = unnorm_weights / sum_weights
        else:
            weights = np.ones(len(valid_cids)) / len(valid_cids)

        # Compute 3D descriptors per valid conformer
        pbf_list, spherocity_list, asphericity_list = [], [], []
        eccentricity_list, inert_list, rog_list = [], [], []

        for cid in valid_cids:
            try:
                pbf_val = float(rdMolDescriptors.CalcPBF(mol_3d, confId=cid))
            except Exception:
                pbf_val = 0.0
            pbf_list.append(pbf_val)

            try:
                sph_val = float(rdMolDescriptors.CalcSpherocityIndex(mol_3d, confId=cid))
            except Exception:
                sph_val = 0.0
            spherocity_list.append(sph_val)

            try:
                asph_val = float(rdMolDescriptors.CalcAsphericity(mol_3d, confId=cid))
            except Exception:
                asph_val = 0.0
            asphericity_list.append(asph_val)

            try:
                ecc_val = float(rdMolDescriptors.CalcEccentricity(mol_3d, confId=cid))
            except Exception:
                ecc_val = 0.0
            eccentricity_list.append(ecc_val)

            try:
                inert_val = float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d, confId=cid))
            except Exception:
                inert_val = 0.0
            inert_list.append(inert_val)

            try:
                rog_val = float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d, confId=cid))
            except Exception:
                rog_val = 0.0
            rog_list.append(rog_val)

        # Boltzmann weighted averages
        b_pbf = float(np.sum(weights * np.array(pbf_list)))
        b_spherocity = float(np.sum(weights * np.array(spherocity_list)))
        b_asphericity = float(np.sum(weights * np.array(asphericity_list)))
        b_eccentricity = float(np.sum(weights * np.array(eccentricity_list)))
        b_inert = float(np.sum(weights * np.array(inert_list)))
        b_rog = float(np.sum(weights * np.array(rog_list)))

        # Ensemble entropy & energy span
        energy_span = float(np.max(delta_e))
        # Shannon conformational entropy: - sum(p_i * ln(p_i))
        entropy = float(-np.sum(weights * np.log(weights + 1e-15)))

        return {
            "boltzmann_pbf": b_pbf,
            "boltzmann_spherocity": b_spherocity,
            "boltzmann_asphericity": b_asphericity,
            "boltzmann_eccentricity": b_eccentricity,
            "boltzmann_inertial_shape_factor": b_inert,
            "boltzmann_radius_of_gyration": b_rog,
            "conformer_energy_span": energy_span,
            "conformational_entropy": entropy,
        }

    except Exception:
        return default_vals


def extract_boltzmann_conformer_vector(
    mol_or_smiles: Union[Chem.Mol, str],
    num_confs: int = 10,
) -> np.ndarray:
    """Extract Boltzmann-weighted 3D conformer ensemble features as a 1D NumPy array."""
    feat_dict = compute_boltzmann_conformer_features(mol_or_smiles, num_confs=num_confs)
    return np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float32)


def batch_extract_boltzmann_features(
    smiles_list: List[str],
    num_confs: int = 10,
    cache_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Extract Boltzmann ensemble features for a list of SMILES with optional caching.

    Args:
        smiles_list: List of SMILES strings.
        num_confs: Number of conformers per molecule.
        cache_path: Optional file path (.npy or .npz) to load/save features.
        verbose: Print progress.

    Returns:
        2D NumPy array of shape (len(smiles_list), 8).
    """
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            if verbose:
                print(f"Loading Boltzmann conformer features from cache: {p}")
            data = np.load(p)
            if isinstance(data, np.lib.npyio.NpzFile):
                return data["features"]
            return data

    if verbose:
        print(
            f"Extracting Boltzmann {num_confs}-conformer 3D features for {len(smiles_list)} molecules..."
        )

    results = []
    for i, s in enumerate(smiles_list):
        vec = extract_boltzmann_conformer_vector(s, num_confs=num_confs)
        results.append(vec)
        if verbose and (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(smiles_list)} compounds...")

    matrix = np.vstack(results).astype(np.float32)

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".npz":
            np.savez_compressed(p, features=matrix)
        else:
            np.save(p, matrix)
        if verbose:
            print(f"Saved Boltzmann 3D features to cache: {p}")

    return matrix
