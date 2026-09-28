"""Unit tests for Pretrained Foundation Encoders (ChemBERTa & ESM-2) in DTI Phase B."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.models.dti.pretrained_encoders import ChemBERTaEncoder, ESM2Encoder


class DummyHFModelOutput:
    """Mock output for HuggingFace Transformers model."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


def _create_mock_hf_model(hidden_size: int = 384):
    """Creates a mock Hugging Face model and tokenizer."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.config = MagicMock()
    mock_model.config.hidden_size = hidden_size

    # Create real dummy parameter so next(model.parameters()).device works
    dummy_param = nn.Parameter(torch.randn(2, 2))
    mock_model.parameters.return_value = iter([dummy_param])

    def mock_forward(input_ids, attention_mask):
        b, seq_len = input_ids.shape
        # Return mock token embeddings
        return DummyHFModelOutput(torch.randn(b, seq_len, hidden_size))

    mock_model.side_effect = mock_forward

    mock_tokenizer = MagicMock()

    def mock_tokenize(texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        b = len(texts)
        seq_len = min(10, max_length)
        return {
            "input_ids": torch.ones(b, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(b, seq_len, dtype=torch.long),
        }

    mock_tokenizer.side_effect = mock_tokenize
    return mock_model, mock_tokenizer


@patch("tdc_studio.models.dti.pretrained_encoders.AutoTokenizer.from_pretrained")
@patch("tdc_studio.models.dti.pretrained_encoders.AutoModel.from_pretrained")
def test_chemberta_encoder_forward(mock_automodel, mock_autotokenizer):
    """Test ChemBERTaEncoder forward pass, projection shape, and device placement."""
    mock_model, mock_tok = _create_mock_hf_model(hidden_size=384)
    mock_automodel.return_value = mock_model
    mock_autotokenizer.return_value = mock_tok

    cfg = {
        "model_name": "DeepChem/ChemBERTa-77M-MTR",
        "hidden_dim": 384,
        "out_dim": 256,
        "max_length": 128,
        "freeze_backbone": False,
    }
    encoder = ChemBERTaEncoder(cfg)

    batch = {
        "drug_smiles_str": [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CN1CCC[C@H]1c2cccnc2",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        ]
    }

    out = encoder(batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 256), f"Expected shape (3, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in ChemBERTaEncoder output"


@patch("tdc_studio.models.dti.pretrained_encoders.AutoTokenizer.from_pretrained")
@patch("tdc_studio.models.dti.pretrained_encoders.AutoModel.from_pretrained")
def test_chemberta_encoder_freeze_unfreeze(mock_automodel, mock_autotokenizer):
    """Test freeze and unfreeze methods on ChemBERTaEncoder."""
    mock_model, mock_tok = _create_mock_hf_model(hidden_size=384)
    mock_automodel.return_value = mock_model
    mock_autotokenizer.return_value = mock_tok

    encoder = ChemBERTaEncoder({"freeze_backbone": True})
    encoder.freeze()
    assert encoder.freeze_backbone is True
    assert encoder.partially_unfrozen is False

    encoder.unfreeze()
    assert encoder.freeze_backbone is False
    assert encoder.partially_unfrozen is False

    encoder.unfreeze(last_n_layers=2)
    assert encoder.freeze_backbone is False
    assert encoder.partially_unfrozen is True
    assert encoder.unfrozen_layers == 2


@patch("tdc_studio.models.dti.pretrained_encoders.AutoTokenizer.from_pretrained")
@patch("tdc_studio.models.dti.pretrained_encoders.AutoModel.from_pretrained")
def test_esm2_encoder_freeze_unfreeze(mock_automodel, mock_autotokenizer):
    """Test freeze, full unfreeze, and partial unfreeze on ESM2Encoder."""
    mock_model, mock_tok = _create_mock_hf_model(hidden_size=480)
    mock_automodel.return_value = mock_model
    mock_autotokenizer.return_value = mock_tok

    encoder = ESM2Encoder({"freeze_backbone": True})
    encoder.freeze()
    assert encoder.freeze_backbone is True
    assert encoder.partially_unfrozen is False

    encoder.unfreeze()
    assert encoder.freeze_backbone is False
    assert encoder.partially_unfrozen is False

    encoder.unfreeze(last_n_layers=2)
    assert encoder.freeze_backbone is False
    assert encoder.partially_unfrozen is True
    assert encoder.unfrozen_layers == 2


@patch("tdc_studio.models.dti.pretrained_encoders.AutoTokenizer.from_pretrained")
@patch("tdc_studio.models.dti.pretrained_encoders.AutoModel.from_pretrained")
def test_esm2_encoder_forward(mock_automodel, mock_autotokenizer):
    """Test ESM2Encoder forward pass and output projection shape."""
    mock_model, mock_tok = _create_mock_hf_model(hidden_size=480)
    mock_automodel.return_value = mock_model
    mock_autotokenizer.return_value = mock_tok

    cfg = {
        "model_name": "facebook/esm2_t12_35M_UR50D",
        "hidden_dim": 480,
        "out_dim": 256,
        "max_length": 512,
        "freeze_backbone": True,
    }
    encoder = ESM2Encoder(cfg)

    batch = {
        "target_seq_str": [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD",
            "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDP",
        ]
    }

    out = encoder(batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 256), f"Expected shape (2, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in ESM2Encoder output"


@patch("tdc_studio.models.dti.pretrained_encoders.AutoTokenizer.from_pretrained")
@patch("tdc_studio.models.dti.pretrained_encoders.AutoModel.from_pretrained")
def test_graph_dta_model_with_foundation_encoders(mock_automodel, mock_autotokenizer):
    """Test GraphDTAModel assembled with ChemBERTa and ESM-2 foundation encoders."""
    mock_model_chem, mock_tok_chem = _create_mock_hf_model(hidden_size=384)
    mock_model_esm, mock_tok_esm = _create_mock_hf_model(hidden_size=480)

    # Alternate side_effects between chem and esm initializations
    mock_automodel.side_effect = [mock_model_chem, mock_model_esm]
    mock_autotokenizer.side_effect = [mock_tok_chem, mock_tok_esm]

    cfg = {
        "task_type": "dta",
        "drug_encoder": {
            "type": "chembert_encoder",
            "model_name": "DeepChem/ChemBERTa-77M-MTR",
            "hidden_dim": 384,
            "out_dim": 256,
        },
        "target_encoder": {
            "type": "esm2_encoder",
            "model_name": "facebook/esm2_t12_35M_UR50D",
            "hidden_dim": 480,
            "out_dim": 256,
        },
        "fusion": {
            "hidden_dim": 512,
            "dropout": 0.1,
            "out_dim": 1,
        },
    }

    model = GraphDTAModel(cfg)

    batch = {
        "drug_smiles_str": ["CC(=O)O", "CCN"],
        "target_seq_str": ["MKTAY", "MSHHW"],
        "labels": torch.tensor([1.2, 3.4], dtype=torch.float32),
    }

    # 1. Feature extraction
    h_drug, h_target = model.extract_features(batch)
    assert h_drug.shape == (2, 256)
    assert h_target.shape == (2, 256)

    # 2. Forward prediction
    preds = model(batch)
    assert preds.shape == (2, 1)
    assert not torch.isnan(preds).any()

    # 3. Loss computation
    loss = model.compute_loss(preds, batch["labels"])
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.ndim == 0
