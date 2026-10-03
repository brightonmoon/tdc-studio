"""Pluggable Building Block Storage Adapters for In-House and Commercial Chemical Databases."""

import abc
import csv
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from rdkit import Chem

logger = logging.getLogger("tdc_studio.retrosynthesis.adapters")


@dataclass
class BuildingBlockRecord:
    """Unified Canonical Chemical Building Block Record."""

    smiles: str
    inchikey: str
    supplier: str = "Commercial Vendor"
    catalog_id: str = "N/A"
    source_type: Literal["internal_stock", "commercial_vendor", "virtual_space"] = "commercial_vendor"
    cost_per_gram: float = 10.0  # USD or KRW normalized per gram
    lead_time_days: int = 2      # 0 for immediate in-house stock, >0 for vendor shipping
    inventory_amount_g: float = 100.0  # Available grams in stock
    purity_percent: float = 95.0
    cas_number: Optional[str] = None
    storage_temp: str = "rt"     # "rt", "4C", "-20C"
    hazards: List[str] = field(default_factory=list)  # e.g. ["air_sensitive", "toxic"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseStockAdapter(abc.ABC):
    """Abstract Base Class for building block database backends."""

    @abc.abstractmethod
    def is_in_stock(self, inchikey: str) -> bool:
        """Check if compound exists in this repository."""
        pass

    @abc.abstractmethod
    def get_record(self, inchikey: str) -> Optional[BuildingBlockRecord]:
        """Fetch full compound record by InChIKey."""
        pass

    @abc.abstractmethod
    def load_records(self, records: List[BuildingBlockRecord]) -> int:
        """Batch load compound records into storage."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class InMemoryStockAdapter(BaseStockAdapter):
    """High-speed in-memory hash index adapter (Tier 0)."""

    def __init__(self, name: str = "InMemory"):
        self.name = name
        self._records: Dict[str, BuildingBlockRecord] = {}

    def is_in_stock(self, inchikey: str) -> bool:
        return inchikey in self._records

    def get_record(self, inchikey: str) -> Optional[BuildingBlockRecord]:
        return self._records.get(inchikey)

    def load_records(self, records: List[BuildingBlockRecord]) -> int:
        count = 0
        for r in records:
            if r.inchikey:
                self._records[r.inchikey] = r
                count += 1
        return count

    def add_smiles(
        self,
        smiles: str,
        cost_per_gram: float = 10.0,
        supplier: str = "CommonStock",
        source_type: Literal["internal_stock", "commercial_vendor", "virtual_space"] = "commercial_vendor",
        lead_time_days: int = 1,
    ) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        canon_smi = Chem.MolToSmiles(mol, canonical=True)
        ikey = Chem.MolToInchiKey(mol)
        rec = BuildingBlockRecord(
            smiles=canon_smi,
            inchikey=ikey,
            supplier=supplier,
            catalog_id=f"CAT-{abs(hash(ikey)) % 1000000:06d}",
            source_type=source_type,
            cost_per_gram=cost_per_gram,
            lead_time_days=lead_time_days,
        )
        self._records[ikey] = rec
        return True

    def __len__(self) -> int:
        return len(self._records)


class SQLiteStockAdapter(BaseStockAdapter):
    """Persistent SQLite database adapter for medium/large catalogs (Tier 1)."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS building_blocks (
                inchikey TEXT PRIMARY KEY,
                smiles TEXT NOT NULL,
                supplier TEXT,
                catalog_id TEXT,
                source_type TEXT,
                cost_per_gram REAL,
                lead_time_days INTEGER,
                inventory_amount_g REAL,
                purity_percent REAL,
                cas_number TEXT,
                storage_temp TEXT,
                hazards TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_smiles ON building_blocks (smiles)")
        self.conn.commit()

    def is_in_stock(self, inchikey: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM building_blocks WHERE inchikey = ? LIMIT 1", (inchikey,))
        return cur.fetchone() is not None

    def get_record(self, inchikey: str) -> Optional[BuildingBlockRecord]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT smiles, inchikey, supplier, catalog_id, source_type,
                   cost_per_gram, lead_time_days, inventory_amount_g,
                   purity_percent, cas_number, storage_temp, hazards
            FROM building_blocks WHERE inchikey = ?
            """,
            (inchikey,),
        )
        row = cur.fetchone()
        if not row:
            return None
        hazards = row[11].split(";") if row[11] else []
        return BuildingBlockRecord(
            smiles=row[0],
            inchikey=row[1],
            supplier=row[2],
            catalog_id=row[3],
            source_type=row[4],
            cost_per_gram=float(row[5]),
            lead_time_days=int(row[6]),
            inventory_amount_g=float(row[7]),
            purity_percent=float(row[8]),
            cas_number=row[9],
            storage_temp=row[10],
            hazards=hazards,
        )

    def load_records(self, records: List[BuildingBlockRecord]) -> int:
        cur = self.conn.cursor()
        inserted = 0
        for r in records:
            hazards_str = ";".join(r.hazards) if r.hazards else ""
            cur.execute(
                """
                INSERT OR REPLACE INTO building_blocks
                (inchikey, smiles, supplier, catalog_id, source_type,
                 cost_per_gram, lead_time_days, inventory_amount_g,
                 purity_percent, cas_number, storage_temp, hazards)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.inchikey,
                    r.smiles,
                    r.supplier,
                    r.catalog_id,
                    r.source_type,
                    r.cost_per_gram,
                    r.lead_time_days,
                    r.inventory_amount_g,
                    r.purity_percent,
                    r.cas_number,
                    r.storage_temp,
                    hazards_str,
                ),
            )
            inserted += 1
        self.conn.commit()
        return inserted

    def __len__(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM building_blocks")
        return cur.fetchone()[0]


class CSVStockAdapter(BaseStockAdapter):
    """File-backed CSV/TSV adapter for procurement and LIMS file exports."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._memory = InMemoryStockAdapter(name=f"CSV_{os.path.basename(file_path)}")
        if os.path.exists(file_path):
            self._load_csv()

    def _load_csv(self):
        delimiter = "\t" if self.file_path.endswith((".tab", ".tsv")) else ","
        records = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                smi = row.get("smiles") or row.get("SMILES") or row.get("reactant") or row.get("structure")
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    continue
                ikey = Chem.MolToInchiKey(mol)
                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                cost = float(row.get("cost_per_gram") or row.get("price") or 10.0)
                supplier = row.get("supplier") or row.get("vendor") or "CSV_Catalog"
                source_type = row.get("source_type") or "commercial_vendor"
                lead_time = int(row.get("lead_time_days") or 3)
                records.append(
                    BuildingBlockRecord(
                        smiles=canon_smi,
                        inchikey=ikey,
                        supplier=supplier,
                        catalog_id=row.get("catalog_id") or f"ID-{abs(hash(ikey)) % 100000}",
                        source_type=source_type,
                        cost_per_gram=cost,
                        lead_time_days=lead_time,
                    )
                )
        self._memory.load_records(records)

    def is_in_stock(self, inchikey: str) -> bool:
        return self._memory.is_in_stock(inchikey)

    def get_record(self, inchikey: str) -> Optional[BuildingBlockRecord]:
        return self._memory.get_record(inchikey)

    def load_records(self, records: List[BuildingBlockRecord]) -> int:
        return self._memory.load_records(records)

    def __len__(self) -> int:
        return len(self._memory)


class UnifiedStockManager:
    """Multi-tier Stock Orchestrator prioritizing in-house inventory over commercial vendors."""

    def __init__(self):
        # Priority-ordered adapter list: (priority_level, adapter)
        self.adapters: List[BaseStockAdapter] = []
        self.banned_inchikeys: set = set()
        self._cache: Dict[str, Optional[BuildingBlockRecord]] = {}

    def register_adapter(self, adapter: BaseStockAdapter, priority: int = 10):
        """Register storage adapter (Lower priority number = queried first)."""
        self.adapters.append(adapter)
        self._cache.clear()

    def ban_compound(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        self.banned_inchikeys.add(Chem.MolToInchiKey(mol))
        return True

    def unban_compound(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        key = Chem.MolToInchiKey(mol)
        if key in self.banned_inchikeys:
            self.banned_inchikeys.remove(key)
            return True
        return False

    def clear_banned(self):
        self.banned_inchikeys.clear()

    def is_in_stock(self, smiles_or_inchikey: str) -> bool:
        """Check if molecule exists in any registered stock adapter and is not banned."""
        ikey = self._resolve_inchikey(smiles_or_inchikey)
        if not ikey or ikey in self.banned_inchikeys:
            return False

        if ikey in self._cache:
            return self._cache[ikey] is not None

        for adapter in self.adapters:
            if adapter.is_in_stock(ikey):
                rec = adapter.get_record(ikey)
                self._cache[ikey] = rec
                return True

        self._cache[ikey] = None
        return False

    def get_record(self, smiles_or_inchikey: str) -> Optional[BuildingBlockRecord]:
        """Fetch building block metadata from highest priority adapter."""
        ikey = self._resolve_inchikey(smiles_or_inchikey)
        if not ikey or ikey in self.banned_inchikeys:
            return None

        if ikey in self._cache and self._cache[ikey] is not None:
            return self._cache[ikey]

        for adapter in self.adapters:
            rec = adapter.get_record(ikey)
            if rec:
                self._cache[ikey] = rec
                return rec
        return None

    def get_cost(self, smiles: str, default: float = 100.0) -> float:
        rec = self.get_record(smiles)
        return rec.cost_per_gram if rec else default

    def get_lead_time(self, smiles: str, default: int = 14) -> int:
        rec = self.get_record(smiles)
        return rec.lead_time_days if rec else default

    @staticmethod
    def _resolve_inchikey(smiles_or_inchikey: str) -> Optional[str]:
        if len(smiles_or_inchikey) == 27 and "-" in smiles_or_inchikey:
            return smiles_or_inchikey
        mol = Chem.MolFromSmiles(smiles_or_inchikey)
        return Chem.MolToInchiKey(mol) if mol else None

    def total_compounds(self) -> int:
        return sum(len(a) for a in self.adapters)
