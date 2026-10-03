"""Comprehensive Test Suite for Phase 3: Pocket-Aware DTI & Active Learning (Tasks 3-1 ~ 3-5)."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Task 3-3
from tdc_studio.active_learning import (
    conformal_expected_improvement,
    conformal_upper_confidence_bound,
    recommend_top_wetlab_candidates,
)

# Task 3-2
from tdc_studio.docking import (
    AutoDockVinaEngine,
    BioNeMoDiffDockEngine,
    DockingResult,
    NeurosnapEngine,
    StructureFetcher,
    TamarindEngine,
    get_docking_engine,
)

# Task 3-1 & Task 3-5
from tdc_studio.features.pocket_extractor import (
    extract_pocket_residue_mask,
    parse_p2rank_predictions,
    slice_pocket_embeddings,
)
from tdc_studio.features.target_attention import (
    analyze_target_attention,
)

# Task 3-4
from tdc_studio.models.dti import (
    FewShotDTAAdapter,
    FewShotTrainer,
    GraphDTAModel,
    ResidualBottleneckAdapter,
)

# =====================================================================
# Task 3-1: Pocket-Specific Slicing & P2Rank Parser Tests
# =====================================================================


def test_p2rank_parser_and_mask():
    mock_p2rank_csv = """name,   rank,   score,  probability,   center_x,   center_y,   center_z,   residue_ids, surf_atom_ids
pocket1,  1,    28.54,        0.842,     12.450,     24.110,     -5.320,  A_745 A_746 A_750 A_790 A_791, 12 14 18
pocket2,  2,    11.20,        0.315,     30.120,     10.500,     15.200,  A_100 A_101 A_105, 55 58
"""
    pockets = parse_p2rank_predictions(mock_p2rank_csv)
    assert len(pockets) == 2
    assert pockets[0]["rank"] == 1
    assert pockets[0]["score"] == 28.54
    assert pockets[0]["probability"] == 0.842
    assert pockets[0]["residue_indices_1based"] == [745, 746, 750, 790, 791]
    assert pockets[0]["residue_indices_0based"] == [744, 745, 749, 789, 790]

    # Test mask extraction
    mask = extract_pocket_residue_mask(
        seq_len=800, pocket_indices=[744, 745, 749, 789, 790], zero_indexed=True
    )
    assert mask.shape == (800,)
    assert mask[744] is True or mask[744] == 1
    assert mask[789] is True or mask[789] == 1
    assert mask[0] is False or mask[0] == 0


def test_slice_pocket_embeddings():
    # Tensor of shape [B=2, L=1000, D=64]
    B, L, D = 2, 1000, 64
    x = torch.randn(B, L, D)
    pocket_idx = [10, 25, 40, 80, 95]

    sliced = slice_pocket_embeddings(x, pocket_idx, zero_indexed=True)
    assert sliced.shape == (B, len(pocket_idx), D)
    # Check values match exact indices
    assert torch.allclose(sliced[:, 0, :], x[:, 10, :])
    assert torch.allclose(sliced[:, 4, :], x[:, 95, :])

    # Slicing 2D tensor [L, D]
    x_2d = torch.randn(L, D)
    sliced_2d = slice_pocket_embeddings(x_2d, pocket_idx, zero_indexed=True)
    assert sliced_2d.shape == (len(pocket_idx), D)
    assert torch.allclose(sliced_2d[1], x_2d[25])


def test_graph_dta_model_with_pocket_indices():
    cfg = {
        "drug_encoder": {
            "type": "protein_cnn",
            "out_dim": 128,
        },  # Use CNN for drug as quick mock encoder
        "target_encoder": {"type": "protein_cnn", "out_dim": 128},
        "fusion": {
            "type": "cross_attention",
            "drug_dim": 128,
            "target_dim": 128,
            "hidden_dim": 128,
            "num_heads": 2,
        },
    }
    # ProteinCNN expects batch["target_seq"]
    model = GraphDTAModel(cfg)
    model.eval()

    # Synthetic batch: 2 proteins of length 300, drug seq of length 40
    batch = {
        "drug_graph": torch.randint(0, 20, (2, 40)),
        "target_seq": torch.randint(0, 20, (2, 300)),
        "pocket_indices": [5, 12, 18, 24, 30, 45, 52, 60],  # 8 pocket residues
    }
    with torch.no_grad():
        out = model(batch, return_sequence=True)
        assert out.shape == (2, 1)


# =====================================================================
# Task 3-2: Multi-Provider 3D Docking Bridge & Fetcher Tests
# =====================================================================


def test_structure_fetcher_routing():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = StructureFetcher(cache_dir=tmpdir)
        # Test routing heuristic without requiring network
        assert fetcher.af_dir.exists()
        assert fetcher.rcsb_dir.exists()

        # Write dummy PDB into cache to verify hit
        dummy_pdb = fetcher.rcsb_dir / "1M17.pdb"
        dummy_pdb.write_text(
            "HEADER    EGFR KINASE DOMAIN\nATOM      1  N   MET A   1\n", encoding="utf-8"
        )

        path, src = fetcher.fetch_structure("1m17")
        assert src == "rcsb"
        assert path == dummy_pdb


def test_docking_engine_factory_and_mock_docking():
    vina_engine = get_docking_engine("vina")
    assert isinstance(vina_engine, AutoDockVinaEngine)

    bionemo_engine = get_docking_engine("bionemo")
    assert isinstance(bionemo_engine, BioNeMoDiffDockEngine)

    neurosnap_engine = get_docking_engine("neurosnap")
    assert isinstance(neurosnap_engine, NeurosnapEngine)

    tamarind_engine = get_docking_engine("tamarind")
    assert isinstance(tamarind_engine, TamarindEngine)

    # Test mock/simulation execution
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(
            b"HEADER TEST PDB\nATOM      1  CA  MET A   1       0.0   0.0   0.0  1.00 90.00           C\n"
        )
        pdb_path = f.name

    try:
        res = vina_engine.dock(
            ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            receptor_path=pdb_path,
            center=(10.0, 15.0, -5.0),
            num_poses=5,
        )
        assert isinstance(res, DockingResult)
        assert res.success is True
        assert res.top_affinity < 0.0  # Physically reasonable negative binding energy
        assert len(res.poses) == 5
        assert res.poses[0].affinity_kcal_mol <= res.poses[1].affinity_kcal_mol

        # Test Cloud DiffDock mock
        res_cloud = bionemo_engine.dock(
            ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
            receptor_path=pdb_path,
            num_poses=3,
        )
        assert res_cloud.success is True
        assert len(res_cloud.poses) == 3
        assert res_cloud.top_affinity < -5.0
    finally:
        Path(pdb_path).unlink(missing_ok=True)


# =====================================================================
# Task 3-3: Active Learning & Top 10 Experiment Recommender Tests
# =====================================================================


def test_conformal_bayesian_acquisition():
    # Expected Improvement
    mu = np.array([8.5, 7.2, 9.1, 8.0])
    q = np.array([0.4, 0.5, 0.3, 0.9])  # Conformal margin
    current_best = 8.6

    ei = conformal_expected_improvement(mu, q, current_best=current_best)
    assert len(ei) == 4
    # Candidate 2 (mu=9.1 > 8.6) should have the highest EI
    assert np.argmax(ei) == 2
    assert ei[2] > ei[1]

    # UCB
    ucb = conformal_upper_confidence_bound(mu, q, beta=1.0)
    assert len(ucb) == 4
    assert np.all(ucb >= mu)


def test_diversity_and_top10_recommender():
    candidates = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",  # Omeprazole
        "c1ccc(cc1)C(=O)O",  # Benzoic acid
        "c1ccccc1O",  # Phenol
        "Cc1ccccc1",  # Toluene
        "CCN(CC)CC",  # Triethylamine
        "c1ccncc1",  # Pyridine
        "c1ncccc1C#N",  # 2-cyanopyridine
        "CCOC(=O)c1ccccc1O",  # Ethyl salicylate
    ]
    preds = [8.5, 8.4, 8.2, 8.1, 8.9, 7.9, 7.8, 7.5, 6.8, 7.1, 7.3, 8.3]
    uncertainties = 0.5

    recs = recommend_top_wetlab_candidates(
        candidate_smiles=candidates,
        predictions=preds,
        uncertainties=uncertainties,
        current_best=8.2,
        top_k=5,
        strategy="ei",
        target_name="EGFR_Kinase",
    )

    assert len(recs) == 5
    assert recs[0].rank == 1
    assert recs[0].synthesis_priority == "Urgent (Top 1-3)"
    assert recs[0].predicted_affinity >= 7.0
    assert recs[0].lower_95 < recs[0].predicted_affinity < recs[0].upper_95
    assert "EGFR_Kinase" in recs[0].rationale
    assert recs[0].min_tanimoto_distance > 0.0


# =====================================================================
# Task 3-4: Few-shot LoRA / Residual Adapter Tests
# =====================================================================


def test_residual_bottleneck_adapter():
    in_dim = 64
    adapter = ResidualBottleneckAdapter(in_dim=in_dim, bottleneck_dim=16, scale=0.5)
    x = torch.randn(4, in_dim)
    # Zero-initialized up projection guarantees initial output == input
    out = adapter(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)


def test_few_shot_adapter_fine_tuning():
    cfg = {
        "drug_encoder": {"type": "protein_cnn", "out_dim": 64},
        "target_encoder": {"type": "protein_cnn", "out_dim": 64},
        "fusion": {
            "type": "cross_attention",
            "drug_dim": 64,
            "target_dim": 64,
            "hidden_dim": 64,
            "num_heads": 2,
        },
    }
    base_model = GraphDTAModel(cfg)
    adapter_model = FewShotDTAAdapter(
        base_model=base_model, bottleneck_dim=16, use_output_delta=True
    )

    # Check parameter freezing: base parameters must not have gradients
    for p in base_model.parameters():
        assert p.requires_grad is False

    trainable_params = adapter_model.get_trainable_parameters()
    assert len(trainable_params) > 0
    total_trainable = sum(p.numel() for p in trainable_params)
    assert total_trainable < 50_000  # Lightweight parameter footprint

    # Create tiny in-house measured assay dataset (16 samples)
    dummy_drug = torch.randint(0, 20, (16, 20))
    dummy_target = torch.randint(0, 20, (16, 50))
    dummy_affinity = torch.tensor(
        [8.0 + 0.1 * i for i in range(16)], dtype=torch.float32
    ).unsqueeze(-1)

    class SmallDataset(Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return {
                "drug_graph": dummy_drug[idx],
                "target_seq": dummy_target[idx],
                "affinity": dummy_affinity[idx],
            }

    loader = DataLoader(SmallDataset(), batch_size=4, shuffle=True)
    trainer = FewShotTrainer(adapter_model=adapter_model, lr=5e-3)
    res = trainer.train_few_shot(dataloader=loader, epochs=5)

    assert res["epochs_run"] > 0
    assert len(res["loss_history"]) > 0

    # Test weight saving & loading
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        adapter_model.save_adapter_weights(ckpt_path)
        assert Path(ckpt_path).stat().st_size > 0
        adapter_model.load_adapter_weights(ckpt_path)
    finally:
        Path(ckpt_path).unlink(missing_ok=True)


# =====================================================================
# Task 3-5: Target Residue Cross-Attention XAI Tests
# =====================================================================


def test_target_attention_analyzer():
    # 2D contact map [L_drug=10, L_target=8]
    contact_map = np.zeros((10, 8), dtype=np.float32)
    # Simulate high interaction at target residues 2 and 5 (0-based)
    contact_map[:, 2] = 0.8
    contact_map[:, 5] = 1.2
    contact_map[:, 0] = 0.2

    target_seq = "ACDEFGHIKLMN"  # 12 residues
    coordinate_map = [740, 741, 745, 750, 789, 790, 795, 800]  # Mapping to EGFR catalytic cleft

    result = analyze_target_attention(
        contact_map=contact_map,
        target_seq=target_seq,
        coordinate_map=coordinate_map,
        top_k=5,
    )

    assert len(result.top_residues) == 5
    # Rank 1 should be column 5 -> coordinate 790
    assert result.top_residues[0].rank == 1
    assert result.top_residues[0].residue_index == 790
    assert result.top_residues[0].normalized_score == 1.0
    assert result.top_residues[0].percent_contribution > 40.0
    # Rank 2 should be column 2 -> coordinate 745
    assert result.top_residues[1].rank == 2
    assert result.top_residues[1].residue_index == 745

    # Check PyMOL command generation
    assert "select binding_pocket, resi" in result.pymol_command
    assert "790" in result.pymol_command
    assert "745" in result.pymol_command

    # Check PyMOL script export
    with tempfile.NamedTemporaryFile(suffix=".pml", delete=False) as f:
        pml_path = f.name
    try:
        script = result.export_pymol_script(pml_path, pdb_file="1M17.pdb")
        assert "load 1M17.pdb" in script
        assert "show sticks" in script
        assert Path(pml_path).stat().st_size > 0
    finally:
        Path(pml_path).unlink(missing_ok=True)
