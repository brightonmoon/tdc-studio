"""Unit tests for MolecularStandardizer and CompoundFilter."""

from rdkit import Chem

from tdc_studio.data.standardizer import MolecularStandardizer
from tdc_studio.features.filters import CompoundFilter


def test_standardizer_salt_removal():
    """Test stripping salts and counterions from drug formulations."""
    standardizer = MolecularStandardizer(remove_salts=True, neutralize=True)

    # Hydrochloride salt of tertiary amine
    salt_smi = "CCN(CC)CC.Cl"
    res = standardizer.standardize(salt_smi)

    assert res.is_valid is True
    assert "Cl" in res.fragments_removed
    assert res.standardized_smiles == "CCN(CC)CC"
    assert res.heavy_atom_count == 7


def test_standardizer_solvent_removal():
    """Test removing DMSO or water co-crystallized with drug."""
    standardizer = MolecularStandardizer(remove_salts=True)

    # Aspirin + DMSO solvate
    solvate_smi = "CC(=O)Oc1ccccc1C(=O)O.CS(=O)C"
    res = standardizer.standardize(solvate_smi)

    assert res.is_valid is True
    assert len(res.fragments_removed) == 1
    # DMSO may be canonicalized as C[S+](C)[O-] or CS(=O)C
    removed_smi = res.fragments_removed[0]
    assert Chem.MolToSmiles(Chem.MolFromSmiles(removed_smi)) in [
        Chem.MolToSmiles(Chem.MolFromSmiles("CS(=O)C")),
        Chem.MolToSmiles(Chem.MolFromSmiles("C[S+](C)[O-]")),
    ]
    assert Chem.MolToSmiles(Chem.MolFromSmiles(res.standardized_smiles)) == Chem.MolToSmiles(
        Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    )


def test_standardizer_neutralization():
    """Test uncharging carboxylate sodium salt into carboxylic acid."""
    standardizer = MolecularStandardizer(remove_salts=True, neutralize=True)

    # Sodium benzoate
    salt_smi = "c1ccccc1C(=O)[O-].[Na+]"
    res = standardizer.standardize(salt_smi)

    assert res.is_valid is True
    assert "[Na+]" in res.fragments_removed or "Na" in str(res.fragments_removed)
    assert res.standardized_smiles == "O=C(O)c1ccccc1"


def test_standardizer_invalid_smiles():
    """Test standardizer robustness against invalid SMILES."""
    standardizer = MolecularStandardizer()

    empty_res = standardizer.standardize("")
    assert empty_res.is_valid is False

    invalid_res = standardizer.standardize("invalid_smiles_string!!!")
    assert invalid_res.is_valid is False
    assert len(invalid_res.warning_messages) > 0


def test_compound_filter_clean_drug():
    """Test that a clean, approved-type drug passes all filters."""
    filter_engine = CompoundFilter()

    # Ibuprofen
    ibuprofen_smi = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    res = filter_engine.evaluate(ibuprofen_smi)

    assert res.is_valid is True
    assert res.pains_passed is True
    assert res.brenk_passed is True
    assert res.ro5_passed is True
    assert res.veber_passed is True
    assert res.passed_all is True
    assert res.properties["mw"] < 300.0


def test_compound_filter_pains_detection():
    """Test detection of PAINS alert (e.g. Rhodanine derivative)."""
    filter_engine = CompoundFilter()

    # Rhodanine PAINS pattern
    rhodanine_smi = "O=C1NC(=S)SC1=Cc2ccccc2"
    res = filter_engine.evaluate(rhodanine_smi)

    assert res.is_valid is True
    assert res.pains_passed is False
    assert len(res.pains_alerts) > 0
    assert res.passed_all is False


def test_compound_filter_brenk_detection():
    """Test detection of reactive Brenk alert (e.g. Acyl halide)."""
    filter_engine = CompoundFilter()

    # Benzoyl chloride (acyl halide)
    benzoyl_chloride_smi = "O=C(Cl)c1ccccc1"
    res = filter_engine.evaluate(benzoyl_chloride_smi)

    assert res.is_valid is True
    assert res.brenk_passed is False
    assert len(res.brenk_alerts) > 0
    assert res.passed_all is False


def test_compound_filter_ro5_violation():
    """Test Lipinski Ro5 multi-violation detection."""
    filter_engine = CompoundFilter(max_ro5_violations=1)

    # Ultra-large molecule with high MW and high LogP violating Ro5
    giant_smi = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O"  # High MW, high LogP
    res = filter_engine.evaluate(giant_smi)

    assert res.is_valid is True
    assert res.ro5_violations >= 1
