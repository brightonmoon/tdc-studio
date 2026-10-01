"""Unit tests for TDC Retrosynthesis, Forward Reaction, and Yields data pipeline."""

import torch
from rdkit import Chem

from tdc_studio.data.reaction import ForwardReactionDataModule
from tdc_studio.data.retrosyn import (
    ReactionTokenizer,
    RetroSynDataModule,
    canonicalize_reaction_smiles,
    get_mock_retrosyn_dataset,
    remove_atom_mapping,
)
from tdc_studio.data.yields import YieldsDataModule


def test_remove_atom_mapping():
    mapped_smiles = "[CH3:1][C:2](=[O:3])[OH:4]"
    cleaned = remove_atom_mapping(mapped_smiles)
    assert cleaned == "CC(=O)O"

    mapped_ring = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[Br:7]"
    cleaned_ring = remove_atom_mapping(mapped_ring)
    assert cleaned_ring == "Brc1ccccc1"


def test_canonicalize_reaction_smiles():
    prod = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1-[c:7]2[cH:8][cH:9][cH:10][cH:11][cH:12]2"
    reactants_a = "OB(O)c1ccccc1.Brc1ccccc1"
    reactants_b = "Brc1ccccc1.OB(O)c1ccccc1"

    clean_prod, clean_r_a = canonicalize_reaction_smiles(prod, reactants_a)
    _, clean_r_b = canonicalize_reaction_smiles(prod, reactants_b)

    assert clean_prod == "c1ccc(-c2ccccc2)cc1"
    # Order should be deterministically sorted
    assert clean_r_a == clean_r_b
    assert "." in clean_r_a


def test_reaction_tokenizer_encode_decode():
    tokenizer = ReactionTokenizer(max_length=64)
    assert tokenizer.vocab_size > 50
    assert "<RX_1>" in tokenizer.vocab
    assert "<RX_10>" in tokenizer.vocab

    smiles = "CC(=O)Nc1ccccc1"
    token_ids, attention_mask = tokenizer.encode(smiles, reaction_type=2, max_length=32)

    assert token_ids.shape == torch.Size([32])
    assert attention_mask.shape == torch.Size([32])
    assert token_ids[0].item() == tokenizer.cls_token_id
    assert token_ids[1].item() == tokenizer.vocab["<RX_2>"]
    assert attention_mask.sum().item() > 0

    decoded = tokenizer.decode(token_ids.tolist())
    # Should reconstruct original SMILES without specials
    assert "CC(=O)Nc1ccccc1" in decoded or "C" in decoded


def test_mock_retrosyn_dataset_integrity():
    df = get_mock_retrosyn_dataset(n_samples=60)
    assert len(df) == 60
    assert "input" in df.columns
    assert "output" in df.columns
    assert "reaction_type" in df.columns

    # Verify RDKit validity of all products and reactants
    for idx, row in df.iterrows():
        prod_mol = Chem.MolFromSmiles(row["input"])
        assert prod_mol is not None, f"Invalid product SMILES: {row['input']}"
        for frag in row["output"].split("."):
            frag_mol = Chem.MolFromSmiles(frag)
            assert frag_mol is not None, f"Invalid reactant fragment: {frag}"

    # Check that multiple reaction classes are present
    classes = df["reaction_type"].unique()
    assert len(classes) >= 5


def test_retrosyn_datamodule_loaders():
    dm = RetroSynDataModule(dataset_name="USPTO-50K", max_length=64, seed=42)
    dm.prepare_data()

    assert "train" in dm.splits
    assert "valid" in dm.splits
    assert "test" in dm.splits

    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=8)
    assert len(train_loader) > 0

    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target_ids" in batch
    assert "target_mask" in batch
    assert "reaction_type" in batch

    assert batch["input_ids"].shape == (8, 64)
    assert batch["target_ids"].shape == (8, 64)
    assert batch["reaction_type"].shape == (8,)


def test_forward_reaction_datamodule():
    dm = ForwardReactionDataModule(dataset_name="USPTO", max_length=64, seed=42)
    dm.prepare_data()

    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=4)
    batch = next(iter(train_loader))

    assert "reactants_smiles" in batch
    assert "product_smiles" in batch
    assert batch["input_ids"].shape == (4, 64)
    assert batch["target_ids"].shape == (4, 64)


def test_yields_datamodule():
    dm = YieldsDataModule(dataset_name="Buchwald-Hartwig", n_bits=512, seed=42)
    dm.prepare_data()

    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=4)
    batch = next(iter(train_loader))

    assert "reactions" in batch
    assert "features" in batch
    assert "yield" in batch
    assert batch["features"].shape == (4, 512)
    assert batch["yield"].shape == (4,)
    # Check yield normalization in [0, 1]
    assert (batch["yield"] >= 0.0).all()
    assert (batch["yield"] <= 1.0).all()
