"""Test new biophysical features: Quaternary Nitrogen, Polar Density, Fraction CSP3."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr
from tdc.single_pred import ADME


def main():
    data = ADME(name="PPBR_AZ").get_data()
    smiles = data["Drug"].tolist()
    y = data["Y"].tolist()

    quat_n_smarts = Chem.MolFromSmarts("[NX4+;!$([NX4+]([O-])=O)]")

    feat_quat_n = []
    feat_tpsa_mw = []
    feat_fsp3 = []
    feat_rot_density = []
    y_clean = []

    for s, val in zip(smiles, y):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        n_qn = len(mol.GetSubstructMatches(quat_n_smarts))
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        tpsa_mw = tpsa / max(1.0, mw)
        fsp3 = Descriptors.FractionCSP3(mol)
        hac = Descriptors.HeavyAtomCount(mol)
        rot_dens = Descriptors.NumRotatableBonds(mol) / max(1.0, hac)

        feat_quat_n.append(n_qn)
        feat_tpsa_mw.append(tpsa_mw)
        feat_fsp3.append(fsp3)
        feat_rot_density.append(rot_dens)
        y_clean.append(val)

    print("=== Feature Correlations with PPBR (%) across 1,614 compounds ===")
    for name, f in [
        ("Quaternary Nitrogen [N+]", feat_quat_n),
        ("TPSA / MolWt (Polar Density)", feat_tpsa_mw),
        ("Fraction CSP3 (3D Non-planarity)", feat_fsp3),
        ("Rotatable Bond Density", feat_rot_density),
    ]:
        r, pr_p = pearsonr(f, y_clean)
        sp, sp_p = spearmanr(f, y_clean)
        print(f"  {name:<35}: Pearson r = {r:+.4f} (p={pr_p:.2e}), Spearman rho = {sp:+.4f}")


if __name__ == "__main__":
    main()
