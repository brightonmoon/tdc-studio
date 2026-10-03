"""Commercial Building Blocks and Reagent Stock Indexer."""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem

from tdc_studio.retrosynthesis.adapters import (
    BaseStockAdapter,
    BuildingBlockRecord,
    InMemoryStockAdapter,
    UnifiedStockManager,
)

logger = logging.getLogger("tdc_studio.retrosynthesis.stock")

# Common organic building blocks across medicinal chemistry (halides, acids, amines, boronics)
COMMON_BUILDING_BLOCKS = [
    # Simple Carboxylic Acids & Chlorides
    "CC(=O)O",
    "CCC(=O)O",
    "CCCC(=O)O",
    "c1ccc(C(=O)O)cc1",
    "Cc1ccc(C(=O)O)cc1",
    "COc1ccc(C(=O)O)cc1",
    "c1ccc(CC(=O)O)cc1",
    "O=C(O)c1ccncc1",
    "O=C(O)c1cccnc1",
    "CC(=O)Cl",
    "c1ccc(C(=O)Cl)cc1",
    "c1ccc(S(=O)(=O)Cl)cc1",
    "CS(=O)(=O)Cl",
    # Primary & Secondary Amines
    "CN",
    "CCN",
    "CCCN",
    "CCCCN",
    "CC(C)N",
    "C1CCNCC1",
    "C1COCCN1",
    "C1CCCN1",
    "c1ccc(N)cc1",
    "Cc1ccc(N)cc1",
    "COc1ccc(N)cc1",
    "c1ccc(CN)cc1",
    "c1ccc(CCN)cc1",
    "NCc1ccccc1",
    "CNc1ccccc1",
    "Nc1ccncc1",
    "Nc1ccccc1N",
    "Nc1ccccc1O",
    # Halides & Alkylating Agents
    "CI",
    "CBr",
    "CCI",
    "CCBr",
    "CCCl",
    "CCCBr",
    "CCCCBr",
    "c1ccc(Br)cc1",
    "c1ccc(I)cc1",
    "c1ccc(Cl)cc1",
    "c1ccc(CBr)cc1",
    "c1ccc(CCl)cc1",
    "Cc1ccc(Br)cc1",
    "COc1ccc(Br)cc1",
    "Brc1ccncc1",
    "Ic1ccccc1",
    # Boronic Acids & Esters (Suzuki precursors)
    "OB(O)c1ccccc1",
    "Cc1ccc(B(O)O)cc1",
    "COc1ccc(B(O)O)cc1",
    "OB(O)c1ccncc1",
    "OB(O)c1cccnc1",
    "OB(O)C1CC1",
    "OB(O)c1ccc(F)cc1",
    "OB(O)c1ccc(Cl)cc1",
    # Alcohols & Phenols
    "CO",
    "CCO",
    "CCCO",
    "CCCCO",
    "CC(C)O",
    "CC(C)(C)O",
    "c1ccc(O)cc1",
    "Cc1ccc(O)cc1",
    "COc1ccc(O)cc1",
    "c1ccc(CO)cc1",
    "OCc1ccncc1",
    # Aldehydes & Ketones
    "C=O",
    "CC=O",
    "CCC=O",
    "c1ccc(C=O)cc1",
    "Cc1ccc(C=O)cc1",
    "COc1ccc(C=O)cc1",
    "CC(=O)C",
    "CC(=O)c1ccccc1",
    "c1ccc(C(=O)c2ccccc2)cc1",
    # Inorganic / Small reagents
    "O",
    "N",
    "S",
    "Cl",
    "Br",
    "C#N",
    "O=CO",
    # Base building blocks
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1cncnc1",
]


class StockLibrary:
    """Tiered Commercial Building Block Library with InChIKey O(1) hash indexing."""

    def __init__(self, load_builtin: bool = True):
        self.inchikey_to_smiles: Dict[str, str] = {}
        self.inchikey_to_cost: Dict[str, float] = {}
        self.inchikey_to_tier: Dict[str, int] = {}
        self.banned_inchikeys: set = set()

        self.stock_manager = UnifiedStockManager()
        self._default_adapter = InMemoryStockAdapter(name="BuiltinStock")
        self.stock_manager.register_adapter(self._default_adapter, priority=5)

        if load_builtin:
            self.load_compounds(COMMON_BUILDING_BLOCKS, default_cost=10.0, tier=0)

    @staticmethod
    def smiles_to_inchikey(smiles: str) -> Optional[str]:
        """Convert SMILES to InChIKey for standard canonical hash lookup."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)

    def register_adapter(self, adapter: BaseStockAdapter, priority: int = 10) -> None:
        """Register an external stock backend (SQLite, CSV, custom LIMS/vendor API)."""
        self.stock_manager.register_adapter(adapter, priority=priority)

    def ban_compound(self, smiles: str) -> bool:
        """Mark a compound as banned / out of stock (e.g. supply chain bottleneck, patent issue)."""
        key = self.smiles_to_inchikey(smiles)
        if key is None:
            return False
        self.banned_inchikeys.add(key)
        self.stock_manager.ban_compound(smiles)
        return True

    def unban_compound(self, smiles: str) -> bool:
        """Remove a compound from the banned blacklist."""
        key = self.smiles_to_inchikey(smiles)
        if key is None or key not in self.banned_inchikeys:
            return False
        self.banned_inchikeys.remove(key)
        self.stock_manager.unban_compound(smiles)
        return True

    def clear_banned(self) -> None:
        """Clear all active bans."""
        self.banned_inchikeys.clear()
        self.stock_manager.clear_banned()

    def is_banned(self, smiles: str) -> bool:
        """Check if molecule is currently banned."""
        key = self.smiles_to_inchikey(smiles)
        return bool(key and key in self.banned_inchikeys)

    def add_compound(
        self,
        smiles: str,
        cost_per_gram: float = 10.0,
        tier: int = 0,
        supplier: str = "CommonStock",
        lead_time_days: int = 1,
    ) -> bool:
        """Register a single compound into the stock database."""
        key = self.smiles_to_inchikey(smiles)
        if key is None:
            return False
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        cost = max(0.1, float(cost_per_gram))
        self.inchikey_to_smiles[key] = canon_smi
        self.inchikey_to_cost[key] = cost
        self.inchikey_to_tier[key] = int(tier)

        self._default_adapter.add_smiles(
            canon_smi,
            cost_per_gram=cost,
            supplier=supplier,
            lead_time_days=lead_time_days,
        )
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
        """Check if molecule exists in commercial stock in O(1) time and is not banned."""
        key = self.smiles_to_inchikey(smiles)
        if key is None or key in self.banned_inchikeys:
            return False
        if key in self.inchikey_to_smiles:
            return True
        return self.stock_manager.is_in_stock(key)

    def get_cost(self, smiles: str, default: float = 100.0) -> float:
        """Retrieve estimated commercial purchase cost per gram ($/g)."""
        key = self.smiles_to_inchikey(smiles)
        if key is None:
            return default
        if key in self.inchikey_to_cost:
            return self.inchikey_to_cost[key]
        return self.stock_manager.get_cost(smiles, default=default)

    def get_record(self, smiles: str) -> Optional[BuildingBlockRecord]:
        """Fetch rich building block metadata (supplier, lead time, hazards)."""
        return self.stock_manager.get_record(smiles)

    def get_lead_time(self, smiles: str, default: int = 2) -> int:
        """Retrieve delivery or procurement lead time in days."""
        return self.stock_manager.get_lead_time(smiles, default=default)

    def check_all_in_stock(self, smiles_list: List[str]) -> Tuple[bool, Dict[str, bool]]:
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
        return max(len(self.inchikey_to_smiles), self.stock_manager.total_compounds())
