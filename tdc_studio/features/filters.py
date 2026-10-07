"""Medicinal Chemistry Filters: PAINS, Brenk, Lipinski Ro5, and Veber Rules.

Uses RDKit FilterCatalog for high-speed substructure filtering of Pan-Assay
Interference Compounds (PAINS) and reactive/toxic Brenk structural alerts,
alongside physicochemical drug-likeness rules.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

logger = logging.getLogger("tdc_studio.features.filters")


@dataclass
class FilterResult:
    """Result of medicinal chemistry filtering."""

    smiles: str
    is_valid: bool
    passed_all: bool
    pains_passed: bool
    pains_alerts: List[str] = field(default_factory=list)
    brenk_passed: bool = True
    brenk_alerts: List[str] = field(default_factory=list)
    ro5_passed: bool = True
    ro5_violations: int = 0
    veber_passed: bool = True
    properties: Dict[str, float] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "smiles": self.smiles,
            "is_valid": self.is_valid,
            "passed_all": self.passed_all,
            "pains_passed": self.pains_passed,
            "pains_alerts": self.pains_alerts,
            "brenk_passed": self.brenk_passed,
            "brenk_alerts": self.brenk_alerts,
            "ro5_passed": self.ro5_passed,
            "ro5_violations": self.ro5_violations,
            "veber_passed": self.veber_passed,
            "properties": {k: round(v, 2) for k, v in self.properties.items()},
            "rejection_reasons": self.rejection_reasons,
        }


class CompoundFilter:
    """High-throughput medicinal chemistry structural and drug-likeness filter.

    Wraps RDKit FilterCatalog for:
    - PAINS (Pan-Assay Interference Compounds, Subsets A, B, C: 480 patterns)
    - Brenk (Unwanted reactive and toxic functional groups: 105 patterns)
    - Lipinski Rule of 5 (Ro5: MW <= 500, LogP <= 5.0, HBD <= 5, HBA <= 10)
    - Veber Oral Bioavailability Rule (Rotatable Bonds <= 10, TPSA <= 140 A^2)
    """

    def __init__(
        self,
        enable_pains: bool = True,
        enable_brenk: bool = True,
        enable_ro5: bool = True,
        enable_veber: bool = True,
        max_ro5_violations: int = 1,
    ):
        """Initialize compound filter catalogs.

        Args:
            enable_pains: Check for PAINS assay interferers.
            enable_brenk: Check for Brenk unwanted reactive groups.
            enable_ro5: Check Lipinski Rule of 5.
            enable_veber: Check Veber rules for oral bioavailability.
            max_ro5_violations: Maximum tolerated Ro5 violations (default 1).
        """
        self.enable_pains = enable_pains
        self.enable_brenk = enable_brenk
        self.enable_ro5 = enable_ro5
        self.enable_veber = enable_veber
        self.max_ro5_violations = max_ro5_violations

        # 1. PAINS catalog (Subsets A, B, C)
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        self.pains_catalog = FilterCatalog(pains_params)

        # 2. Brenk catalog
        brenk_params = FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog(brenk_params)

    def evaluate(self, mol_or_smiles: Union[str, Chem.Mol]) -> FilterResult:
        """Evaluate a molecule against all enabled medicinal chemistry filters.

        Args:
            mol_or_smiles: SMILES string or RDKit Mol object.

        Returns:
            FilterResult containing detailed pass/fail status and triggered alerts.
        """
        mol = None
        smi = ""

        if isinstance(mol_or_smiles, str):
            smi = mol_or_smiles.strip()
            if not smi:
                return FilterResult(
                    smiles="",
                    is_valid=False,
                    passed_all=False,
                    pains_passed=False,
                    rejection_reasons=["Empty SMILES string"],
                )
            mol = Chem.MolFromSmiles(smi)
        elif isinstance(mol_or_smiles, Chem.Mol):
            mol = mol_or_smiles
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = "<MolObject>"

        if mol is None:
            return FilterResult(
                smiles=smi,
                is_valid=False,
                passed_all=False,
                pains_passed=False,
                rejection_reasons=["Invalid molecule structure"],
            )

        rejection_reasons = []
        pains_alerts = []
        brenk_alerts = []

        # 1. PAINS check
        pains_passed = True
        if self.enable_pains:
            matches = self.pains_catalog.GetMatches(mol)
            if matches:
                pains_passed = False
                for match in matches:
                    pains_alerts.append(match.GetDescription())
                rejection_reasons.append(
                    f"Failed PAINS filter ({len(pains_alerts)} alert(s): {', '.join(pains_alerts[:3])})"
                )

        # 2. Brenk check
        brenk_passed = True
        if self.enable_brenk:
            matches = self.brenk_catalog.GetMatches(mol)
            if matches:
                brenk_passed = False
                for match in matches:
                    brenk_alerts.append(match.GetDescription())
                rejection_reasons.append(
                    f"Failed Brenk filter ({len(brenk_alerts)} alert(s): {', '.join(brenk_alerts[:3])})"
                )

        # 3. Calculate physicochemical properties
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rot_bonds = int(Lipinski.NumRotatableBonds(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))

        props = {
            "mw": mw,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rot_bonds,
            "tpsa": tpsa,
        }

        # 4. Lipinski Ro5 check
        ro5_violations = 0
        if mw > 500.0:
            ro5_violations += 1
        if logp > 5.0:
            ro5_violations += 1
        if hbd > 5:
            ro5_violations += 1
        if hba > 10:
            ro5_violations += 1

        ro5_passed = ro5_violations <= self.max_ro5_violations
        if self.enable_ro5 and not ro5_passed:
            rejection_reasons.append(
                f"Failed Lipinski Rule of 5 ({ro5_violations} violations > max {self.max_ro5_violations})"
            )

        # 5. Veber Rule check
        veber_passed = (rot_bonds <= 10) and (tpsa <= 140.0)
        if self.enable_veber and not veber_passed:
            reasons = []
            if rot_bonds > 10:
                reasons.append(f"Rotatable bonds {rot_bonds} > 10")
            if tpsa > 140.0:
                reasons.append(f"TPSA {tpsa:.1f} > 140")
            rejection_reasons.append(f"Failed Veber oral bioavailability rule ({', '.join(reasons)})")

        passed_all = pains_passed and brenk_passed and ro5_passed and veber_passed

        return FilterResult(
            smiles=smi,
            is_valid=True,
            passed_all=passed_all,
            pains_passed=pains_passed,
            pains_alerts=pains_alerts,
            brenk_passed=brenk_passed,
            brenk_alerts=brenk_alerts,
            ro5_passed=ro5_passed,
            ro5_violations=ro5_violations,
            veber_passed=veber_passed,
            properties=props,
            rejection_reasons=rejection_reasons,
        )

    def evaluate_batch(
        self, items: List[Union[str, Chem.Mol]]
    ) -> List[FilterResult]:
        """Evaluate a batch of molecules."""
        return [self.evaluate(item) for item in items]
