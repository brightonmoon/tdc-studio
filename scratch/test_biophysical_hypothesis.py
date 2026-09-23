"""Statistical hypothesis verification of physiological pH 7.4 ionization & HSA/AAG binding motifs on TDC PPBR_AZ."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from scipy.stats import pearsonr, spearmanr
from tdc.single_pred import ADME


def compute_biophysical_descriptors(s: str) -> dict:
    mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
    if mol is None:
        return {}

    # 1. Acidic Functional Groups (HSA binding anchors)
    carboxylic_acid_patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    sulfonamide_patt = Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#7X3H1,H2;!$(NC=O)])")
    tetrazole_patt = Chem.MolFromSmarts("c1nnn[nH]1")
    phenol_patt = Chem.MolFromSmarts("[OX2H][cX3]")
    phosphate_patt = Chem.MolFromSmarts("[PX4](=O)([OX2H,OX1-])([OX2H,OX1-])")

    n_cooh = len(mol.GetSubstructMatches(carboxylic_acid_patt))
    n_sulfonamide = len(mol.GetSubstructMatches(sulfonamide_patt))
    n_tetrazole = len(mol.GetSubstructMatches(tetrazole_patt))
    n_phenol = len(mol.GetSubstructMatches(phenol_patt))
    n_phosphate = len(mol.GetSubstructMatches(phosphate_patt))

    # Henderson-Hasselbalch fractions for acids at pH 7.4:
    # COOH (pKa ~ 4.2): 1 / (1 + 10^(4.2 - 7.4)) = 0.999
    # Sulfonamide (pKa ~ 6.0): 1 / (1 + 10^(6.0 - 7.4)) = 0.962
    # Tetrazole (pKa ~ 4.8): 0.997
    # Phenol (pKa ~ 9.8): 1 / (1 + 10^(9.8 - 7.4)) = 0.004
    f_anion = (
        n_cooh * 0.999
        + n_sulfonamide * 0.962
        + n_tetrazole * 0.997
        + n_phosphate * 1.99
        + n_phenol * 0.004
    )

    # 2. Basic Functional Groups (AAG binding anchors)
    aliphatic_amine_patt = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS(=O)=O);!$(Nc)]")
    aromatic_amine_patt = Chem.MolFromSmarts("[NX3;H2,H1;$(Nc)]")
    guanidine_patt = Chem.MolFromSmarts("[NX3][CX3](=[NX2])")
    pyridine_patt = Chem.MolFromSmarts("[nX2;$(n1ccccc1)]")
    piperazine_patt = Chem.MolFromSmarts("[NX3]1CCNCC1")

    n_aliphatic_amine = len(mol.GetSubstructMatches(aliphatic_amine_patt))
    n_aromatic_amine = len(mol.GetSubstructMatches(aromatic_amine_patt))
    n_guanidine = len(mol.GetSubstructMatches(guanidine_patt))
    n_pyridine = len(mol.GetSubstructMatches(pyridine_patt))
    n_piperazine = len(mol.GetSubstructMatches(piperazine_patt))

    # Henderson-Hasselbalch fractions for bases at pH 7.4:
    # Aliphatic amine (pKa ~ 9.8): 1 / (1 + 10^(7.4 - 9.8)) = 0.996
    # Guanidine (pKa ~ 12.5): 0.9999
    # Piperazine (pKa ~ 8.2): 0.863
    # Pyridine (pKa ~ 5.2): 0.006
    # Aromatic amine (pKa ~ 4.6): 0.002
    f_cation = (
        n_aliphatic_amine * 0.996
        + n_guanidine * 0.9999
        + n_piperazine * 0.863
        + n_pyridine * 0.006
        + n_aromatic_amine * 0.002
    )

    # Net formal charge at physiological pH 7.4
    q_net_74 = f_cation - f_anion
    is_anion_74 = 1.0 if f_anion >= 0.5 else 0.0
    is_cation_74 = 1.0 if f_cation >= 0.5 else 0.0
    is_neutral_74 = 1.0 if (f_anion < 0.5 and f_cation < 0.5) else 0.0
    is_zwitterion_74 = 1.0 if (f_anion >= 0.5 and f_cation >= 0.5) else 0.0

    # 3. Sudlow Site I Motif (HSA Warfarin site: bulky hydrophobic + anionic anchor)
    n_aromatic_rings = Descriptors.NumAromaticRings(mol)
    mol_wt = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)

    sudlow_site_1 = 1.0 if (n_aromatic_rings >= 2 and (n_cooh + n_sulfonamide + n_tetrazole >= 1 or logp >= 3.0)) else 0.0

    # 4. Sudlow Site II Motif (HSA Ibuprofen site: aromatic + carboxylate/anion tail)
    sudlow_site_2 = 1.0 if (n_cooh >= 1 and n_aromatic_rings >= 1 and logp >= 1.5) else 0.0

    # 5. AAG Motif (lipophilic cation: logP > 2.0 + basic nitrogen)
    aag_motif = 1.0 if (f_cation >= 0.5 and logp >= 2.0) else 0.0

    # 6. Physiological Lipophilicity (logD at pH 7.4 approximation)
    if is_anion_74:
        logd_74 = logp - np.log10(1.0 + 10 ** (7.4 - 4.2))
    elif is_cation_74:
        logd_74 = logp - np.log10(1.0 + 10 ** (9.5 - 7.4))
    else:
        logd_74 = logp

    return {
        "f_anion": float(f_anion),
        "f_cation": float(f_cation),
        "q_net_74": float(q_net_74),
        "is_anion_74": float(is_anion_74),
        "is_cation_74": float(is_cation_74),
        "is_neutral_74": float(is_neutral_74),
        "is_zwitterion_74": float(is_zwitterion_74),
        "sudlow_site_1": float(sudlow_site_1),
        "sudlow_site_2": float(sudlow_site_2),
        "aag_motif": float(aag_motif),
        "logd_74": float(logd_74),
        "n_cooh": float(n_cooh),
        "n_sulfonamide": float(n_sulfonamide),
        "n_aliphatic_amine": float(n_aliphatic_amine),
    }


def main():
    print("Loading TDC PPBR_AZ dataset...")
    data = ADME(name="PPBR_AZ")
    df = data.get_data()
    print(f"Loaded {len(df)} compounds.")

    records = []
    y_raw = df["Y"].values
    # Gibbs logit
    fb = np.clip(y_raw / 100.0, 1e-4, 1.0 - 1e-4)
    y_logit = np.log(fb / (1.0 - fb))

    for idx, row in df.iterrows():
        smiles = row["Drug"]
        desc = compute_biophysical_descriptors(smiles)
        records.append(desc)

    desc_df = pd.DataFrame(records)

    print("\n=== Statistical Hypothesis Validation: Physiological pH 7.4 & HSA/AAG Descriptors ===")
    print(f"{'Descriptor':<22} | {'Pearson r (vs Logit)':<20} | {'Spearman rho':<15} | {'p-value':<12}")
    print("-" * 75)

    for col in desc_df.columns:
        vals = desc_df[col].values
        r, p_val = pearsonr(vals, y_logit)
        rho, _ = spearmanr(vals, y_raw)
        print(f"{col:<22} | r = {r:+0.4f} (p={p_val:.2e}) | rho = {rho:+0.4f}")

    # Sub-population analysis
    print("\n=== Biophysical Sub-cohort Binding Distributions ===")
    anion_mask = desc_df["is_anion_74"] == 1.0
    cation_mask = desc_df["is_cation_74"] == 1.0
    neutral_mask = desc_df["is_neutral_74"] == 1.0
    s1_mask = desc_df["sudlow_site_1"] == 1.0
    s2_mask = desc_df["sudlow_site_2"] == 1.0
    aag_mask = desc_df["aag_motif"] == 1.0

    print(f"Anionic (HSA candidates, N={anion_mask.sum()}): Mean PPBR = {y_raw[anion_mask].mean():.2f}% ± {y_raw[anion_mask].std():.2f}%")
    print(f"Cationic (AAG candidates, N={cation_mask.sum()}): Mean PPBR = {y_raw[cation_mask].mean():.2f}% ± {y_raw[cation_mask].std():.2f}%")
    print(f"Neutral compounds (N={neutral_mask.sum()}): Mean PPBR = {y_raw[neutral_mask].mean():.2f}% ± {y_raw[neutral_mask].std():.2f}%")
    print(f"Sudlow Site I Motif (N={s1_mask.sum()}): Mean PPBR = {y_raw[s1_mask].mean():.2f}% (Logit mean = {y_logit[s1_mask].mean():.2f})")
    print(f"Sudlow Site II Motif (N={s2_mask.sum()}): Mean PPBR = {y_raw[s2_mask].mean():.2f}% (Logit mean = {y_logit[s2_mask].mean():.2f})")
    print(f"AAG Lipophilic Cation (N={aag_mask.sum()}): Mean PPBR = {y_raw[aag_mask].mean():.2f}%")


if __name__ == "__main__":
    main()
