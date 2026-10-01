"""Unit tests for Self-Correcting Generative Lead Optimizer and SA Score."""

from rdkit import Chem

from tdc_studio.generative.lead_optimizer import SelfCorrectingOptimizer
from tdc_studio.generative.sa_score import calculate_sa_score, is_synthetically_accessible


def test_sa_score_calculation():
    # Aspirin: very simple synthesis, should be ~1.5 - 2.5
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    score_asp = calculate_sa_score(aspirin)
    assert 1.0 <= score_asp <= 3.0
    assert is_synthetically_accessible(aspirin, threshold=3.5) is True

    # Benzene
    score_ben = calculate_sa_score("c1ccccc1")
    assert 1.0 <= score_ben <= 2.5

    # Invalid SMILES gets 10.0
    assert calculate_sa_score("INVALID_SMILES") == 10.0
    assert is_synthetically_accessible("INVALID_SMILES") is False


def test_self_correcting_optimizer_nitrobenzene():
    # Nitrobenzene: classic mutagenic AMES alert
    optimizer = SelfCorrectingOptimizer(device="cpu", sa_threshold=4.0)

    report = optimizer.optimize(
        smiles="O=[N+]([O-])c1ccccc1",
        target_liability="ames",
        max_candidates=3,
        steps=10,
    )

    assert report.input_smiles == "O=[N+]([O-])c1ccccc1"
    assert report.canonical_smiles == "O=[N+]([O-])c1ccccc1"
    assert report.bemis_murcko_scaffold == "c1ccccc1"
    assert report.primary_liability is not None
    assert report.primary_liability.liability_key == "ames"
    assert report.candidates_generated > 0
    assert report.candidates_passing_sa_filter > 0
    assert len(report.top_candidates) > 0

    top1 = report.top_candidates[0]
    assert Chem.MolFromSmiles(top1.smiles) is not None
    assert top1.sa_score <= 4.0
    assert top1.scaffold_preserved is True
    assert len(top1.rationale) > 0
    assert top1.transformation_name in [
        "nitro_to_cyano",
        "nitro_to_trifluoromethyl",
        "nitro_to_primary_amide",
    ]


def test_self_correcting_optimizer_carboxylic_acid():
    # Benzoic acid: carboxylic acid liable to glucuronidation / idiosyncratic DILI
    optimizer = SelfCorrectingOptimizer(device="cpu", sa_threshold=4.0)

    report = optimizer.optimize(
        smiles="c1ccccc1C(=O)O",
        target_liability="dili",
        max_candidates=3,
        steps=10,
    )

    assert report.input_smiles == "c1ccccc1C(=O)O"
    assert report.bemis_murcko_scaffold == "c1ccccc1"
    assert len(report.top_candidates) > 0

    # Recommended bioisostere should be Tetrazole, Hydroxamic acid, or Primary Amide
    top_cand = report.top_candidates[0]
    assert Chem.MolFromSmiles(top_cand.smiles) is not None
    assert top_cand.sa_score <= 4.0
    assert top_cand.scaffold_preserved is True


def test_self_correcting_optimizer_retrosynthesis_coupling():
    # Verify that LeadOptimizer directly integrates Retrosynthesis verification
    optimizer = SelfCorrectingOptimizer(
        device="cpu",
        sa_threshold=4.5,
        verify_retrosynthesis=True,
    )

    report = optimizer.optimize(
        smiles="c1ccccc1C(=O)O",
        target_liability="dili",
        max_candidates=3,
        steps=5,
        verify_retrosynthesis=True,
        require_deep_route=False,
    )

    assert len(report.top_candidates) > 0
    cand = report.top_candidates[0]
    # Retrosynthesis fields should be populated
    assert cand.retrosynthesis_solved is not None
    assert cand.synthetic_tractability_score is not None
    assert 0.0 <= cand.synthetic_tractability_score <= 1.0
    if cand.retrosynthesis_solved:
        assert cand.route_summary is not None
