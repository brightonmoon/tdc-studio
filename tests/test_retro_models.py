"""Unit tests for Phase 2: Retrosynthesis, Forward Verification, and Yield Models."""

import torch
from rdkit import Chem

from tdc_studio.models.retrosynthesis.forward_verifier import ForwardVerifier
from tdc_studio.models.retrosynthesis.hybrid_policy import HybridRetroPolicy
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.models.retrosynthesis.seq2seq_retro import Seq2SeqRetroModel
from tdc_studio.models.retrosynthesis.yield_predictor import YieldPredictor


def test_rule_retro_policy_amide():
    policy = RuleRetroPolicy()
    # N-methylbenzamide -> Benzoic acid + Methylamine
    product = "c1ccc(C(=O)NC)cc1"
    candidates = policy.predict_reactants(product, top_k=5)

    assert len(candidates) > 0
    reactants, score = candidates[0]
    assert score > 0.0
    # Must be valid RDKit molecules
    for frag in reactants.split("."):
        mol = Chem.MolFromSmiles(frag)
        assert mol is not None, f"Invalid reactant fragment: {frag}"

    # Carboxylic acid and amine should be present
    assert "C(=O)O" in reactants or "NC" in reactants or "CN" in reactants


def test_rule_retro_policy_suzuki():
    policy = RuleRetroPolicy()
    # Biphenyl -> Bromobenzene + Phenylboronic acid
    product = "c1ccc(-c2ccccc2)cc1"
    candidates = policy.predict_reactants(product, top_k=5)

    assert len(candidates) > 0
    reactants, score = candidates[0]
    assert "B" in reactants or "Br" in reactants


def test_seq2seq_retro_model_forward():
    model = Seq2SeqRetroModel(
        {
            "d_model": 64,
            "nhead": 2,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 128,
            "max_length": 32,
        }
    )

    batch_size = 4
    seq_len = 32
    vocab_size = model.vocab_size

    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    batch = {
        "input_ids": src,
        "target_ids": tgt,
    }

    out = model(batch)
    assert "loss" in out
    assert "logits" in out
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0


def test_hybrid_retro_policy():
    neural = Seq2SeqRetroModel(
        {
            "d_model": 64,
            "nhead": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "max_length": 32,
        }
    )
    rule = RuleRetroPolicy()
    hybrid = HybridRetroPolicy(neural_model=neural, rule_policy=rule)

    product = "c1ccc(C(=O)NC)cc1"
    candidates = hybrid.predict_reactants(product, top_k=3)
    assert len(candidates) > 0
    for react, score in candidates:
        assert isinstance(react, str)
        assert score > 0.0


def test_forward_verifier():
    verifier = ForwardVerifier()

    # Amide coupling round-trip: Benzoic acid + Methylamine -> N-methylbenzamide
    reactants = "c1ccc(C(=O)O)cc1.CN"
    expected = "c1ccc(C(=O)NC)cc1"

    is_valid, conf = verifier.verify_reaction(reactants, expected)
    assert is_valid is True
    assert conf >= 0.8

    # Non-canonical SMILES representation of the same molecule should succeed
    alt_expected = "CNC(=O)c1ccccc1"
    is_valid_alt, conf_alt = verifier.verify_reaction(reactants, alt_expected)
    assert is_valid_alt is True
    assert conf_alt >= 0.8

    # Incompatible reactants should fail verification
    bad_reactants = "c1ccccc1.c1ccccc1"
    is_valid_bad, conf_bad = verifier.verify_reaction(bad_reactants, expected)
    assert is_valid_bad is False
    assert conf_bad == 0.0


def test_yield_predictor():
    predictor = YieldPredictor({"hidden_dim": 64, "n_bits": 512})

    reactants = "c1ccc(Br)cc1.c1ccc(N)cc1"
    product = "c1ccc(Nc2ccccc2)cc1"

    yield_val = predictor.predict_yield(reactants, product)
    assert isinstance(yield_val, float)
    assert 0.0 <= yield_val <= 100.0

    # Test tensor forward
    features = torch.randn(4, 512)
    preds = predictor(features)
    assert preds.shape == (4,)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
