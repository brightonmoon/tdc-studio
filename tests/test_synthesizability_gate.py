"""Unit tests for Phase 5: SynthesizabilityGate and Lead Optimizer integration."""


from tdc_studio.generative.synthesizability_gate import SynthesizabilityGate


def test_synthesizability_gate_tier1_reject():
    gate = SynthesizabilityGate(sa_threshold=2.0)
    # Cubane / strained polycyclic cage molecule has high SAScore (~4.5+)
    strained_mol = "C12C3C4C1C5C2C3C45"
    report = gate.evaluate_candidate(strained_mol)

    assert report.passed is False
    assert report.tier1_sa_passed is False
    assert "SAScore" in (report.rejection_reason or "")


def test_synthesizability_gate_tier2_pass():
    gate = SynthesizabilityGate(sa_threshold=4.5)
    # N-methylbenzamide is simple and decomposes into 2 stock reagents in 1 step
    mol = "c1ccc(C(=O)NC)cc1"
    report = gate.evaluate_candidate(mol, require_deep_route=False)

    assert report.tier1_sa_passed is True
    assert report.tier2_1step_passed is True
    assert report.passed is True


def test_synthesizability_gate_tier3_deep_route():
    gate = SynthesizabilityGate(sa_threshold=4.5)
    mol = "c1ccc(C(=O)NC)cc1"
    report = gate.evaluate_candidate(mol, require_deep_route=True)

    assert report.passed is True
    assert report.tier3_route_solved is True
    assert report.route is not None
    assert report.route.solved is True
    assert report.route.total_depth >= 1
