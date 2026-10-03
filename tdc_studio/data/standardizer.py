"""Molecular Standardization & Salt/Solvent Stripping Module.

Ensures high-integrity chemical representations for downstream ADMET, DTI,
and Generative pipelines by removing counterions (salts), solvents, neutralizing
charges, and standardizing tautomers/stereocenters via RDKit rdMolStandardize.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger("tdc_studio.data.standardizer")


@dataclass
class StandardizationResult:
    """Result of molecular standardization and fragment cleanup."""

    original_smiles: str
    standardized_smiles: str
    is_valid: bool
    fragments_removed: List[str] = field(default_factory=list)
    was_neutralized: bool = False
    molecular_weight: float = 0.0
    logp: float = 0.0
    heavy_atom_count: int = 0
    warning_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "original_smiles": self.original_smiles,
            "standardized_smiles": self.standardized_smiles,
            "is_valid": self.is_valid,
            "fragments_removed": self.fragments_removed,
            "was_neutralized": self.was_neutralized,
            "molecular_weight": round(self.molecular_weight, 2),
            "logp": round(self.logp, 2),
            "heavy_atom_count": self.heavy_atom_count,
            "warnings": self.warning_messages,
        }


class MolecularStandardizer:
    """RDKit rdMolStandardize-based standardization engine.

    Features:
    1. Fragment Chooser: Isolates parent drug scaffold from counterions (HCl, TFA, Na+, etc.) and solvents.
    2. Uncharger: Neutralizes formal charges into stable non-ionic representations where possible.
    3. Normalizer & Cleanup: Corrects non-standard valence representations and cleans aromaticity.
    4. Canonicalization: Produces canonical, isomeric RDKit SMILES.
    """

    def __init__(
        self,
        remove_salts: bool = True,
        neutralize: bool = True,
        canonicalize_tautomer: bool = False,
    ):
        """Initialize standardizer with specified processing flags.

        Args:
            remove_salts: Choose largest organic fragment (strips salts & solvents).
            neutralize: Uncharge formal ions into neutral parent.
            canonicalize_tautomer: Resolve canonical tautomeric state.
        """
        self.remove_salts = remove_salts
        self.neutralize = neutralize
        self.canonicalize_tautomer = canonicalize_tautomer

        # RDKit standardizer components
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        self.uncharger = rdMolStandardize.Uncharger()
        if self.canonicalize_tautomer:
            self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def standardize(self, mol_or_smiles: Union[str, Chem.Mol]) -> StandardizationResult:
        """Standardize a single molecule or SMILES string.

        Args:
            mol_or_smiles: SMILES string or RDKit Mol object.

        Returns:
            StandardizationResult with cleaned SMILES and diagnostic metadata.
        """
        orig_smi = ""
        mol = None
        warnings = []

        if isinstance(mol_or_smiles, str):
            orig_smi = mol_or_smiles.strip()
            if not orig_smi:
                return StandardizationResult(
                    original_smiles="",
                    standardized_smiles="",
                    is_valid=False,
                    warning_messages=["Empty SMILES string provided"],
                )
            try:
                mol = Chem.MolFromSmiles(orig_smi)
            except Exception as e:
                warnings.append(f"RDKit parse exception: {str(e)}")
        elif isinstance(mol_or_smiles, Chem.Mol):
            mol = Chem.Mol(mol_or_smiles)  # copy
            try:
                orig_smi = Chem.MolToSmiles(mol)
            except Exception:
                orig_smi = "<MolObject>"

        if mol is None:
            return StandardizationResult(
                original_smiles=orig_smi,
                standardized_smiles="",
                is_valid=False,
                warning_messages=warnings or ["Invalid SMILES or unable to parse molecule"],
            )

        fragments_removed = []
        was_neutralized = False

        try:
            # 1. Basic RDKit MolStandardize cleanup
            mol = rdMolStandardize.Cleanup(mol)

            # 2. Strip salts & solvents (Largest Fragment Chooser)
            if self.remove_salts:
                # Detect detached fragments
                frags = Chem.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    frag_smiles = [Chem.MolToSmiles(f) for f in frags]
                    mol_parent = self.fragment_chooser.choose(mol)
                    parent_smi = Chem.MolToSmiles(mol_parent)
                    for fs in frag_smiles:
                        if fs != parent_smi:
                            fragments_removed.append(fs)
                    mol = mol_parent

            # 3. Uncharge (Neutralize zwitterions/ionized salts)
            if self.neutralize:
                before_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
                mol = self.uncharger.uncharge(mol)
                after_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
                if before_charge != after_charge:
                    was_neutralized = True

            # 4. Optional tautomer canonicalization
            if self.canonicalize_tautomer:
                mol = self.tautomer_enumerator.Canonicalize(mol)

            # 5. Final canonical SMILES generation
            final_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()

            return StandardizationResult(
                original_smiles=orig_smi,
                standardized_smiles=final_smiles,
                is_valid=True,
                fragments_removed=fragments_removed,
                was_neutralized=was_neutralized,
                molecular_weight=mw,
                logp=logp,
                heavy_atom_count=heavy_atoms,
                warning_messages=warnings,
            )

        except Exception as e:
            logger.warning("Standardization failed for '%s': %s", orig_smi, str(e))
            return StandardizationResult(
                original_smiles=orig_smi,
                standardized_smiles="",
                is_valid=False,
                warning_messages=[f"Standardization error: {str(e)}"],
            )

    def standardize_batch(
        self, items: List[Union[str, Chem.Mol]]
    ) -> List[StandardizationResult]:
        """Standardize a batch of molecules/SMILES strings."""
        return [self.standardize(item) for item in items]
