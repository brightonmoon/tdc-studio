"""Unit tests for BindingPocketExtractor and active pocket window extraction."""

from tdc_studio.features.pocket_extractor import BindingPocketExtractor


def test_pocket_extractor_short_sequence():
    extractor = BindingPocketExtractor()
    seq = "MKWVTFISLLLLFSSAYSRG"
    res = extractor.extract_pocket_window(seq, max_window_len=256)

    assert res["pocket_seq"] == seq
    assert res["is_truncated"] is False
    assert len(res["coordinate_map"]) == len(seq)
    assert res["start_idx"] == 0
    assert res["end_idx"] == len(seq)


def test_pocket_extractor_catalytic_motif_centering():
    extractor = BindingPocketExtractor()
    # Construct a long sequence of 1,200 AA with a kinase catalytic loop motif placed around index 750
    # Motif: [LIVMFYC][HY].D[LIVMFY]K..N[LIVMFYC]{3} -> e.g. "IHRDLKPENLLL"
    prefix = "A" * 740
    motif = "IHRDLKPENLLL"
    suffix = "G" * (1200 - 740 - len(motif))
    long_seq = prefix + motif + suffix
    assert len(long_seq) == 1200

    res = extractor.extract_pocket_window(long_seq, max_window_len=256)

    assert res["is_truncated"] is True
    assert res["method"] == "catalytic_motif"
    assert len(res["pocket_seq"]) == 256
    # The motif must be inside the extracted pocket sequence!
    assert motif in res["pocket_seq"]
    assert res["start_idx"] <= 740
    assert res["end_idx"] >= 740 + len(motif)


def test_pocket_extractor_synthetic_pdb_cavity():
    extractor = BindingPocketExtractor()
    # Synthetic mini PDB with 5 residues and CA atoms
    pdb_lines = [
        "ATOM      1  N   ALA A   1      11.104  13.205  -9.843  1.00 85.00           N",
        "ATOM      2  CA  ALA A   1      11.504  14.205  -8.843  1.00 85.00           C",
        "ATOM      3  CA  GLY A   2      12.504  14.205  -8.843  1.00 90.00           C",
        "ATOM      4  CA  LEU A   3      13.504  14.205  -8.843  1.00 92.00           C",
        "ATOM      5  CA  VAL A   4      14.504  14.205  -8.843  1.00 60.00           C",  # low pLDDT
    ]
    pdb_text = "\n".join(pdb_lines)

    indices = extractor.extract_structure_cavity_indices(pdb_text, top_k=2, min_plddt=70.0)
    assert len(indices) <= 2
    # Residue 4 (index 3) should be filtered out by pLDDT < 70
    assert 3 not in indices
