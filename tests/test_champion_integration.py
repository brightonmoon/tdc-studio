import os

from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline


def test_champion_models_loading_and_inference():
    export_dir = "models/export"
    pipeline = UnifiedADMETPipeline.from_exported_directory(export_dir, device="cpu")

    # Verify champions are loaded if present in models/export
    if os.path.exists(os.path.join(export_dir, "ames_champion", "ames_model.joblib")):
        assert pipeline.ames_champion is not None, "AMES champion model should be loaded"

    if os.path.exists(os.path.join(export_dir, "clearance_cascade", "clearance_cascade_model.joblib")):
        assert pipeline.clearance_cascade is not None, "Clearance cascade model should be loaded"

    if os.path.exists(os.path.join(export_dir, "lipophilicity_stacker", "lipo_stacker_model.joblib")):
        assert pipeline.lipo_stacker is not None, "Lipophilicity stacker should be loaded"

    # Run inference on test molecule (Aspirin: CC(=O)Oc1ccccc1C(=O)O)
    res = pipeline.predict_single("CC(=O)Oc1ccccc1C(=O)O")
    assert res is not None
    assert "lipophilicity_astrazeneca" in res.absorption
    assert "clearance_hepatocyte_az" in res.excretion
    assert "ames" in res.toxicity
    assert "herg" in res.toxicity

    lipo_item = res.absorption["lipophilicity_astrazeneca"]
    cl_item = res.excretion["clearance_hepatocyte_az"]
    ames_item = res.toxicity["ames"]
    herg_item = res.toxicity["herg"]

    assert lipo_item.value is not None
    assert cl_item.value is not None
    assert 0.0 <= ames_item.probability <= 1.0
    assert 0.0 <= herg_item.probability <= 1.0

    print("\n--- Model Inference Sanity Check ---")
    print("Aspirin Lipophilicity (Stacker):", lipo_item.value, lipo_item.decision)
    print("Aspirin Hepatocyte Cl (Cascade):", cl_item.value, cl_item.decision)
    print("Aspirin AMES Mutagenicity (Champion GBDT):", ames_item.probability, ames_item.decision)
    print("Aspirin hERG Cardiotox (Champion D-MPNN):", herg_item.probability, herg_item.decision)
