"""24-dimensional Biophysical Motif Feature Extractor for Lipophilicity (logD7.4).

Captures thermodynamic partition and pH 7.4 ionization drivers:
1. Crippen MolLogP
2. MolMR (Molar Refractivity)
3. LabuteASA (Approximate Surface Area)
4. TPSA (Topological Polar Surface Area)
5. TPSA_to_ASA_ratio
6. FractionCSP3 (Fraction of sp3 carbons)
7. NumRotatableBonds
8. NumAromaticRings
9. NumAliphaticRings
10. NumSaturatedRings
11. NumHBD (Hydrogen Bond Donors)
12. NumHBA (Hydrogen Bond Acceptors)
13. HBD_to_HBA_ratio
14. Num_F (Fluorine count)
15. Num_Cl (Chlorine count)
16. Num_Br (Bromine count)
17. Num_I (Iodine count)
18. Num_CF3 (Trifluoromethyl count)
19. Num_COOH (Carboxylic acid count - acidic pKa driver)
20. Num_Aliphatic_Amine (Basic pKa driver)
21. Num_Aromatic_Amine
22. Num_Sulfonamide
23. Num_Tetrazole
24. HeavyAtomCount
"""

from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

LIPO_MOTIF_NAMES: List[str] = [
    "MolLogP",
    "MolMR",
    "LabuteASA",
    "TPSA",
    "TPSA_to_ASA_ratio",
    "FractionCSP3",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHBD",
    "NumHBA",
    "HBD_to_HBA_ratio",
    "Num_F",
    "Num_Cl",
    "Num_Br",
    "Num_I",
    "Num_CF3",
    "Num_COOH",
    "Num_Aliphatic_Amine",
    "Num_Aromatic_Amine",
    "Num_Sulfonamide",
    "Num_Tetrazole",
    "HeavyAtomCount",
]

# Pre-compiled SMARTS for specific functional groups
SMARTS_PATTERNS = {
    "cf3": Chem.MolFromSmarts("[CX4](F)(F)F"),
    "cooh": Chem.MolFromSmarts("C(=O)[OH]"),
    "aliphatic_amine": Chem.MolFromSmarts("[NX3;!$(NC=O);!$(N=O);!$(Na);!$(N#*);!$(N~[#7])][CX4]"),
    "aromatic_amine": Chem.MolFromSmarts("c[NX3;H2,H1,$(N([#6])[#6])]"),
    "sulfonamide": Chem.MolFromSmarts("S(=O)(=O)[NX3]"),
    "tetrazole": Chem.MolFromSmarts("c1nnn[nH]1"),
}


class LipoMotifExtractor:
    """Extracts 24-dimensional biophysical and ionization motif vector for logD prediction."""

    def __init__(self):
        self._dim = len(LIPO_MOTIF_NAMES)
        self._names = list(LIPO_MOTIF_NAMES)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def feature_names(self) -> List[str]:
        return list(self._names)

    def extract(self, mol: Optional[Chem.Mol]) -> np.ndarray:
        """Extract 24-dim feature vector from RDKit Mol object."""
        vec = np.zeros(self._dim, dtype=np.float32)
        if mol is None:
            return vec

        try:
            # 1. MolLogP & MolMR
            logp = float(Descriptors.MolLogP(mol))
            mr = float(Descriptors.MolMR(mol))
            vec[0] = logp
            vec[1] = mr

            # 2. Surface areas & ratios
            labute_asa = float(rdMolDescriptors.CalcLabuteASA(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            vec[2] = labute_asa
            vec[3] = tpsa
            vec[4] = float(tpsa / max(labute_asa, 1e-3))

            # 3. Shape & flexibility
            vec[5] = float(rdMolDescriptors.CalcFractionCSP3(mol))
            vec[6] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))

            # 4. Ring counts
            ri = mol.GetRingInfo()
            num_rings = ri.NumRings()
            num_aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
            num_saturated_rings = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
            num_aliphatic_rings = float(num_rings - num_aromatic_rings)
            vec[7] = num_aromatic_rings
            vec[8] = max(0.0, num_aliphatic_rings)
            vec[9] = num_saturated_rings

            # 5. H-bonding
            hbd = float(rdMolDescriptors.CalcNumHBD(mol))
            hba = float(rdMolDescriptors.CalcNumHBA(mol))
            vec[10] = hbd
            vec[11] = hba
            vec[12] = float(hbd / max(hba, 1.0))

            # 6. Halogens
            f_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
            cl_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
            br_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 35)
            i_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 53)
            vec[13] = float(f_count)
            vec[14] = float(cl_count)
            vec[15] = float(br_count)
            vec[16] = float(i_count)

            # 7. Functional groups and ionization triggers
            cf3_matches = len(mol.GetSubstructMatches(SMARTS_PATTERNS["cf3"])) if SMARTS_PATTERNS["cf3"] else 0
            cooh_matches = len(mol.GetSubstructMatches(SMARTS_PATTERNS["cooh"])) if SMARTS_PATTERNS["cooh"] else 0
            aliph_amine_matches = len(mol.GetSubstructMatches(SMARTS_PATTERNS["aliphatic_amine"])) if SMARTS_PATTERNS["aliphatic_amine"] else 0
            arom_amine_matches = len(mol.GetSubstructMatches(SMARTS_PATTERNS["aromatic_amine"])) if SMARTS_PATTERNS["aromatic_amine"] else 0
            sulfon_matches = len(mol.GetSubstructMatches(SMARTS_PATTERNS["sulfonamide"])) if SMARTS_PATTERNS["sulfonamide"] else 0
            tetra_matches = len(mol.GetSubstructMatches(SMARTS_PATTERNS["tetrazole"])) if SMARTS_PATTERNS["tetrazole"] else 0

            vec[17] = float(cf3_matches)
            vec[18] = float(cooh_matches)
            vec[19] = float(aliph_amine_matches)
            vec[20] = float(arom_amine_matches)
            vec[21] = float(sulfon_matches)
            vec[22] = float(tetra_matches)

            # 8. Heavy atom count
            vec[23] = float(mol.GetNumHeavyAtoms())

        except Exception:
            pass

        return vec

    def extract_from_smiles(self, smiles: str) -> np.ndarray:
        """Extract 24-dim feature vector directly from a SMILES string."""
        if not smiles:
            return np.zeros(self._dim, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles)
            return self.extract(mol)
        except Exception:
            return np.zeros(self._dim, dtype=np.float32)


_GLOBAL_LIPO_EXTRACTOR: Optional[LipoMotifExtractor] = None


def get_lipo_motif_extractor() -> LipoMotifExtractor:
    """Singleton getter for LipoMotifExtractor."""
    global _GLOBAL_LIPO_EXTRACTOR
    if _GLOBAL_LIPO_EXTRACTOR is None:
        _GLOBAL_LIPO_EXTRACTOR = LipoMotifExtractor()
    return _GLOBAL_LIPO_EXTRACTOR
