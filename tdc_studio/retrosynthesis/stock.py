"""Commercial Building Blocks and Reagent Stock Indexer."""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger("tdc_studio.retrosynthesis.stock")

# Common organic building blocks across medicinal chemistry (halides, acids, amines, boronics)
COMMON_BUILDING_BLOCKS = [
    # Simple Carboxylic Acids & Chlorides
    "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "c1ccc(C(=O)O)cc1", "Cc1ccc(C(=O)O)cc1",
    "COc1ccc(C(=O)O)cc1", "c1ccc(CC(=O)O)cc1", "O=C(O)c1ccncc1", "O=C(O)c1cccnc1",
    "CC(=O)Cl", "c1ccc(C(=O)Cl)cc1", "c1ccc(S(=O)(=O)Cl)cc1", "CS(=O)(=O)Cl",
    # Primary & Secondary Amines
    "CN", "CCN", "CCCN", "CCCCN", "CC(C)N", "C1CCNCC1", "C1COCCN1", "C1CCCN1",
    "c1ccc(N)cc1", "Cc1ccc(N)cc1", "COc1ccc(N)cc1", "c1ccc(CN)cc1", "c1ccc(CCN)cc1",
    "NCc1ccccc1", "CNc1ccccc1", "Nc1ccncc1", "Nc1ccccc1N", "Nc1ccccc1O",
    # Halides & Alkylating Agents
    "CI", "CBr", "CCI", "CCBr", "CCCl", "CCCBr", "CCCCBr",
    "c1ccc(Br)cc1", "c1ccc(I)cc1", "c1ccc(Cl)cc1", "c1ccc(CBr)cc1", "c1ccc(CCl)cc1",
    "Cc1ccc(Br)cc1", "COc1ccc(Br)cc1", "Brc1ccncc1", "Ic1ccccc1",
    # Boronic Acids & Esters (Suzuki precursors)
    "OB(O)c1ccccc1", "Cc1ccc(B(O)O)cc1", "COc1ccc(B(O)O)cc1", "OB(O)c1ccncc1",
    "OB(O)c1cccnc1", "OB(O)C1CC1", "OB(O)c1ccc(F)cc1", "OB(O)c1ccc(Cl)cc1",
    # Alcohols & Phenols
    "CO", "CCO", "CCCO", "CCCCO", "CC(C)O", "CC(C)(C)O", "c1ccc(O)cc1",
    "Cc1ccc(O)cc1", "COc1ccc(O)cc1", "c1ccc(CO)cc1", "OCc1ccncc1",
    # Aldehydes & Ketones
    "C=O", "CC=O", "CCC=O", "c1ccc(C=O)cc1", "Cc1ccc(C=O)cc1", "COc1ccc(C=O)cc1",
    "CC(=O)C", "CC(=O)c1ccccc1", "c1ccc(C(=O)c2ccccc2)cc1",
    # Inorganic / Small reagents
    "O", "N", "S", "Cl", "Br", "C#N", "O=CO",
    # Base building blocks
    "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1cncnc1",
]


class StockLibrary:
    """Tiered Commercial Building Block Library with InChIKey O(1) hash indexing."""

    def __init__(self, load_builtin: bool = True):
        self.inchikey_to_smiles: Dict[str, str] = {}
        self.inchikey_to_cost: Dict[str, float] = {}
        self.inchikey_to_tier: Dict[str, int] = {}

        if load_builtin:
            self.load_compounds(COMMON_BUILDING_BLOCKS, default_cost=10.0, tier=0)

    @staticmethod
    def smiles_to_inchikey(smiles: str) -> Optional[str]:
        """Convert SMILES to InChIKey for standard canonical hash lookup."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)

    def add_compound(
        self, smiles: str, cost_per_gram: float = 10.0, tier: int = 0
    ) -> bool:
        """Register a single compound into the stock database."""
        key = self.smiles_to_inchikey(smiles)
        if key is None:
            return False
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        self.inchikey_to_smiles[key] = canon_smi
        self.inchikey_to_cost[key] = max(0.1, float(cost_per_gram))
        self.inchikey_to_tier[key] = int(tier)
        return True

    def load_compounds(
        self,
        smiles_iter: Iterable[str],
        default_cost: float = 10.0,
        tier: int = 0,
    ) -> int:
        """Batch load a collection of compound SMILES into stock."""
        loaded = 0
        for smi in smiles_iter:
            if self.add_compound(smi, cost_per_gram=default_cost, tier=tier):
                loaded += 1
        return loaded

    def is_in_stock(self, smiles: str) -> bool:
        """Check if molecule exists in commercial stock in O(1) time."""
        key = self.smiles_to_inchikey(smiles)
        if key is None:
            return False
        return key in self.inchikey_to_smiles

    def get_cost(self, smiles: str, default: float = 100.0) -> float:
        """Retrieve estimated commercial purchase cost per gram ($/g)."""
        key = self.smiles_to_inchikey(smiles)
        if key is None:
            return default
        return self.inchikey_to_cost.get(key, default)

    def check_all_in_stock(
        self, smiles_list: List[str]
    ) -> Tuple[bool, Dict[str, bool]]:
        """Verify whether all precursor molecules in a reaction step are in stock."""
        status = {}
        all_present = True
        for smi in smiles_list:
            present = self.is_in_stock(smi)
            status[smi] = present
            if not present:
                all_present = False
        return all_present, status

    def __len__(self) -> int:
        return len(self.inchikey_to_smiles)
