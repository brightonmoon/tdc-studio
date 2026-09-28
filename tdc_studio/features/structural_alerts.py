"""Ashby-Tennant 100-dimensional Substructure Alert Module for AMES Mutagenicity.

References:
- Ashby, J. & Tennant, R. W. (1988). Chemical structure, Salmonella mutagenicity
  and rodent carcinogenicity: A study of 222 chemicals. Mutation Research, 204(1), 17-115.
- Ashby, J. & Tennant, R. W. (1991). Definitive relationships among chemical structure,
  carcinogenicity and mutagenicity for 301 chemicals. Mutation Research, 257(3), 229-306.
- Hansen, K. et al. (2009). Benchmark data set for in silico prediction of AMES
  mutagenicity. J. Chem. Inf. Model., 49(9), 2077-2081.
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem

# 100 Codified Ashby-Tennant Genotoxic & Mutagenic SMARTS Patterns (Validated RDKit SMARTS)
ASHBY_TENNANT_ALERTS: List[Tuple[str, str]] = [
    # 1. Epoxides, Aziridines & 3-4 Membered Reactive Heterocycles (10)
    ("epoxide_aliphatic", "[C]1[O][C]1"),
    ("epoxide_terminal", "[CH2]1[O][CH]1"),
    ("arene_oxide", "c1c2c(c1)O2"),
    ("aziridine", "[C]1[N][C]1"),
    ("aziridine_n_substituted", "[C]1[N]([#6])[C]1"),
    ("episulfide", "[C]1[S][C]1"),
    ("oxaziridine", "[C]1[O][N]1"),
    ("beta_lactone", "[C]1[C](=O)[O][C]1"),
    ("beta_sultone", "[C]1[S](=O)(=O)[O][C]1"),
    ("beta_lactam", "[C]1[C](=O)[N][C]1"),

    # 2. Nitro & Nitroso Groups (13)
    ("aromatic_nitro", "c[N+](=O)[O-]"),
    ("aliphatic_nitro", "[CX4][N+](=O)[O-]"),
    ("nitro_on_heteroaromatic", "[n][N+](=O)[O-]"),
    ("nitro_vinyl", "[C]=[C][N+](=O)[O-]"),
    ("dinitrophenyl", "c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1"),
    ("trinitrophenyl", "c1([N+](=O)[O-])cc([N+](=O)[O-])cc([N+](=O)[O-])c1"),
    ("aromatic_nitroso", "c[N]=O"),
    ("aliphatic_nitroso", "[CX4][N]=O"),
    ("n_nitroso_dialkyl", "[N]([CX4])([CX4])[N]=O"),
    ("n_nitroso_cyclic", "[N]1([N]=O)[#6]~[#6]~[#6]~[#6]1"),
    ("n_nitroso_aryl", "c[N]([#6])[N]=O"),
    ("n_nitrosourea", "[#6][N]([N]=O)C(=O)[N]"),
    ("nitrosocarbamate", "[N]([N]=O)C(=O)O[#6]"),

    # 3. Alkyl Halides & Halogenated Aliphatics (13)
    ("aliphatic_halide_primary", "[CH2][Cl,Br,I]"),
    ("aliphatic_halide_secondary", "[CH1]([#6])[Cl,Br,I]"),
    ("aliphatic_halide_tertiary", "[CX4]([#6])([#6])([#6])[Cl,Br,I]"),
    ("geminal_dihalide", "[CX4]([Cl,Br,I])[Cl,Br,I]"),
    ("trihalomethyl", "[CX4]([Cl,Br,I])([Cl,Br,I])[Cl,Br,I]"),
    ("haloalkene_vinyl", "[C]=[C][Cl,Br,I]"),
    ("allyl_halide", "[C]=[C]-[CX4][Cl,Br,I]"),
    ("benzyl_halide", "c-[CX4][Cl,Br,I]"),
    ("bis_haloethyl_amine", "[N](CC[Cl,Br,I])CC[Cl,Br,I]"),
    ("haloethyl_amine_nitrogen_mustard", "[N](CC[Cl,Br])"),
    ("haloethyl_sulfide_sulfur_mustard", "[S](CC[Cl,Br])"),
    ("haloalkyl_ether", "[CX4]([Cl,Br,I])O[#6]"),
    ("alpha_halocarbonyl", "C(=O)[CX4][Cl,Br,I]"),

    # 4. Aromatic Amines & Derivatives (12)
    ("primary_aromatic_amine", "c[NH2]"),
    ("secondary_aromatic_amine", "c[NH1][CX4]"),
    ("tertiary_aromatic_amine", "c[N]([CX4])[CX4]"),
    ("aromatic_hydroxylamine", "c[NH][OH]"),
    ("aromatic_n_methylol", "c[NH]CO"),
    ("polycyclic_aromatic_amine", "c1ccc2c(c1)ccc3c2cccc3[NH2]"),
    ("biphenyl_amine", "c1ccc(cc1)-c2ccc([NH2])cc2"),
    ("stilbene_amine", "c1ccc(cc1)/C=C/c2ccc([NH2])cc2"),
    ("heterocyclic_amine_carboline", "c12[nH]c3ccccc3c1c(N)ccn2"),
    ("heterocyclic_amine_imidazo", "c1nc2ccccc2n1[NH2]"),
    ("aminoazo_dye", "c[NH2].[#6]N=N[#6]"),
    ("aromatic_amine_ortho_methyl", "c1c(C)c([NH2])ccc1"),

    # 5. Hydrazines, Azo, Diazo & Azoxy (11)
    ("hydrazine_unsubstituted", "[#6][NH][NH2]"),
    ("hydrazine_1_1_dialkyl", "[#6][N]([#6])[NH2]"),
    ("hydrazine_1_2_dialkyl", "[#6][NH][NH][#6]"),
    ("aryl_hydrazine", "c[NH][NH2]"),
    ("azo_group", "[#6][N]=[N][#6]"),
    ("azo_aromatic", "c[N]=[N]c"),
    ("azoxy_group", "[#6][N]=[N+]([O-])[#6]"),
    ("diazo_group", "[#6]=[N+]=[N-]"),
    ("diazonium", "c[N+]#[N]"),
    ("triazene", "[#6][N]=[N][NH][#6]"),
    ("azide", "[#6][N]=[N+]=[N-]"),

    # 6. Alkylating Sulfonates, Sulfates, Phosphates (8)
    ("alkyl_sulfonate", "[CX4]OS(=O)(=O)[#6]"),
    ("alkyl_sulfate", "[CX4]OS(=O)(=O)O[#6]"),
    ("methyl_sulfonate", "COS(=O)(=O)[#6]"),
    ("ethyl_sulfonate", "CCOS(=O)(=O)[#6]"),
    ("sultone_5_or_6", "[C]1[C]~[C]~[C][S](=O)(=O)O1"),
    ("mustard_sulfonium", "[S+]([#6])(CC[Cl,Br])"),
    ("dialkyl_phosphonate", "P(=O)(O[CX4])(O[CX4])"),
    ("phosphoric_acid_ester_halo", "P(=O)(O[CX4])([Cl,Br,I])"),

    # 7. Alpha, Beta-Unsaturated Carbonyls & Michael Acceptors (12)
    ("alpha_beta_unsaturated_aldehyde", "[CH1](=O)[C]=[C]"),
    ("alpha_beta_unsaturated_ketone", "[#6]C(=O)[C]=[C]"),
    ("alpha_beta_unsaturated_ester", "O=C(O[#6])[C]=[C]"),
    ("alpha_beta_unsaturated_amide", "O=C(N)[C]=[C]"),
    ("alpha_beta_unsaturated_nitrile", "N#C[C]=[C]"),
    ("acrolein_derivative", "C=C-C=O"),
    ("acrylonitrile", "C=CC#N"),
    ("acrylamide", "C=CC(=O)N"),
    ("maleimide", "O=C1C=CC(=O)N1"),
    ("benzoquinone", "O=C1C=CC(=O)C=C1"),
    ("naphthoquinone", "O=C1C=CC(=O)c2ccccc12"),
    ("anthraquinone", "O=C1c2ccccc2C(=O)c3ccccc13"),

    # 8. Polycyclic Aromatic Hydrocarbons (PAHs) & Intercalators (9)
    ("phenanthrene_skeleton", "c1ccc2c(c1)ccc3ccccc23"),
    ("anthracene_skeleton", "c1ccc2cc3ccccc3cc2c1"),
    ("pyrene_skeleton", "c1cc2cccc3ccc4cccc1c4c32"),
    ("chrysene_skeleton", "c1ccc2c(c1)ccc3c2ccc4ccccc34"),
    ("benzopyrene_skeleton", "c1ccc2c(c1)c3cccc4c3c(cc2)cc5ccccc45"),
    ("benzanthracene_skeleton", "c1ccc2c(c1)ccc3cc4ccccc4cc23"),
    ("acridine_skeleton", "c1ccc2nc3ccccc3cc2c1"),
    ("quinolines_isoquinolines", "c1ccc2ncccc2c1"),
    ("carboline_skeleton", "c1ccc2c(c1)[nH]c3ccncc23"),

    # 9. Reactive Carbonyls, Anhydrides & Acylating Agents (7)
    ("acyl_halide", "C(=O)[Cl,Br,I]"),
    ("carboxylic_anhydride", "C(=O)OC(=O)"),
    ("isocyanate", "[#6]N=C=O"),
    ("isothiocyanate", "[#6]N=C=S"),
    ("carbamoyl_halide", "NC(=O)[Cl,Br]"),
    ("chloroformate", "OC(=O)[Cl]"),
    ("aliphatic_aldehyde", "[CX4][CH]=O"),

    # 10. Peroxides, Thioureas & Sulfur Reactive (5)
    ("organic_peroxide", "[#6]OO[#6]"),
    ("organic_hydroperoxide", "[#6]OO[#1]"),
    ("peroxy_acid", "C(=O)OO[#1]"),
    ("thiourea", "N-C(=S)-N"),
    ("dithiocarbamate", "N-C(=S)-S"),
]


class AshbyTennantAlertExtractor:
    """Extracts 100-dimensional binary and count substructure alert vectors for mutagenicity.

    Each bit corresponds to a distinct electrophilic, DNA-reactive, or mutagenic motif
    established in the Ashby-Tennant and Hansen et al. benchmarks.
    """

    def __init__(self):
        self._alert_defs: List[Tuple[str, Chem.Mol]] = []
        for name, smarts in ASHBY_TENNANT_ALERTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                self._alert_defs.append((name, pattern))
            else:
                raise ValueError(f"Invalid SMARTS pattern for '{name}': {smarts}")

        self._feature_names = [name for name, _ in self._alert_defs]
        self._dim = len(self._alert_defs)

    @property
    def dim(self) -> int:
        """Dimension of the alert bitvector (fixed at exactly 100)."""
        return self._dim

    @property
    def feature_names(self) -> List[str]:
        """List of human-readable alert names."""
        return list(self._feature_names)

    def extract(self, mol: Optional[Chem.Mol], return_counts: bool = False) -> np.ndarray:
        """Extract a 100-dim alert vector from an RDKit Mol object.

        Args:
            mol: RDKit Mol instance or None.
            return_counts: If True, returns match counts; otherwise binary (0.0 or 1.0).

        Returns:
            np.ndarray of shape (100,), dtype=float32.
        """
        vec = np.zeros(self._dim, dtype=np.float32)
        if mol is None:
            return vec

        for idx, (_, pattern) in enumerate(self._alert_defs):
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                vec[idx] = float(len(matches)) if return_counts else 1.0

        return vec

    def extract_from_smiles(self, smiles: str, return_counts: bool = False) -> np.ndarray:
        """Extract a 100-dim alert vector directly from a SMILES string."""
        if not smiles:
            return np.zeros(self._dim, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles)
            return self.extract(mol, return_counts=return_counts)
        except Exception:
            return np.zeros(self._dim, dtype=np.float32)

    def get_matched_alerts(self, mol: Optional[Chem.Mol]) -> List[str]:
        """Get the names of all structural alerts triggered by the molecule."""
        if mol is None:
            return []
        matched = []
        for name, pattern in self._alert_defs:
            if mol.HasSubstructMatch(pattern):
                matched.append(name)
        return matched


_GLOBAL_EXTRACTOR: Optional[AshbyTennantAlertExtractor] = None


def get_ashby_tennant_extractor() -> AshbyTennantAlertExtractor:
    """Singleton getter for AshbyTennantAlertExtractor."""
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is None:
        _GLOBAL_EXTRACTOR = AshbyTennantAlertExtractor()
    return _GLOBAL_EXTRACTOR
