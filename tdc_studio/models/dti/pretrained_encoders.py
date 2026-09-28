"""Pretrained Foundation Model Encoders for Drug-Target Interaction (Phase B).

Provides:
- ChemBERTaEncoder: Pretrained molecular language model (DeepChem/ChemBERTa-77M-MTR)
  producing 384-dim (or projected out_dim) compound representations.
- ESM2Encoder: Pretrained protein language model (facebook/esm2_t12_35M_UR50D or t6_8M)
  producing evolutionary/structural protein representations (projected to out_dim).

Design Principles:
- Drop-in replacement for GINE (drug) and ProteinCNN (target) in GraphDTAModel.
- Supports both feature extraction (frozen backbone) and fine-tuning (staged unfreezing).
- Masked mean pooling over non-padding tokens.
- Graceful handling of device placement and dynamic batch structures.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from tdc_studio.core.registry import MODELS
from tdc_studio.data.transforms import AminoAcidTokenizer


@MODELS.register("chembert_encoder")
@MODELS.register("chemberta_encoder")
class ChemBERTaEncoder(nn.Module):
    """Pretrained ChemBERTa compound encoder for molecular SMILES.

    Architecture:
        SMILES str -> ChemBERTa Tokenizer -> RoBERTa Body (384-dim)
                   -> Masked Mean Pool -> Linear Projection -> out_dim (256-dim)

    Args:
        config: Dict containing:
            model_name      : HuggingFace model identifier (default: "DeepChem/ChemBERTa-77M-MTR")
            hidden_dim      : ChemBERTa representation dim (default: 384)
            out_dim         : Projected output dim matching Fusion head (default: 256)
            max_length      : Max token sequence length for truncation (default: 256)
            pooling         : Pooling mode, "mean" or "cls" (default: "mean")
            freeze_backbone : Whether to freeze RoBERTa weights initially (default: False)
            dropout         : Dropout rate after projection (default: 0.1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}
        self.model_name = cfg.get("model_name", "DeepChem/ChemBERTa-77M-MTR")
        self.hidden_dim = cfg.get("hidden_dim", 384)
        self.out_dim = cfg.get("out_dim", 256)
        self.max_length = cfg.get("max_length", 256)
        self.pooling = cfg.get("pooling", "mean")
        self.freeze_backbone = cfg.get("freeze_backbone", False)
        self.partially_unfrozen = False
        self.unfrozen_layers = 0
        self.return_sequence = cfg.get("return_sequence", False)
        dropout = cfg.get("dropout", 0.1)

        # Load HuggingFace tokenizer and transformer body
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        actual_hidden = getattr(self.model.config, "hidden_size", self.hidden_dim)
        self.hidden_dim = actual_hidden

        # Projection layer to match target encoder / fusion head dimensions
        if self.out_dim != self.hidden_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj = nn.Identity()

        if self.freeze_backbone:
            self.freeze()

    def freeze(self) -> None:
        """Freeze all backbone parameters for feature extraction."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.freeze_backbone = True
        self.partially_unfrozen = False
        self.unfrozen_layers = 0

    def unfreeze(self, last_n_layers: Optional[int] = None) -> None:
        """Unfreeze backbone parameters (optionally only the last N transformer layers)."""
        if last_n_layers is None:
            for param in self.model.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            self.partially_unfrozen = False
            self.unfrozen_layers = 0
        else:
            # First freeze all
            self.freeze()
            # Unfreeze pooler/head
            if hasattr(self.model, "pooler") and self.model.pooler is not None:
                for param in self.model.pooler.parameters():
                    param.requires_grad = True
            # Unfreeze last N encoder layers
            encoder_layers = getattr(getattr(self.model, "encoder", None), "layer", None)
            if encoder_layers is not None:
                for layer in encoder_layers[-last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            self.freeze_backbone = False
            self.partially_unfrozen = True
            self.unfrozen_layers = last_n_layers

    def _mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked mean pooling over non-padding tokens."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_smiles(
        self,
        smiles_list: List[str],
        device: torch.device,
        return_sequence: Optional[bool] = None,
    ) -> torch.Tensor:
        """Tokenize and encode a list of SMILES strings.

        Args:
            smiles_list: List of valid SMILES strings.
            device: Target torch device.
            return_sequence: If True, returns full token sequence [B, L, out_dim].

        Returns:
            FloatTensor [B, out_dim] or [B, L, out_dim]
        """
        if return_sequence is None:
            return_sequence = self.return_sequence

        # Sanitize empty strings
        clean_smiles = [s if (s and isinstance(s, str)) else "C" for s in smiles_list]

        encoded = self.tokenizer(
            clean_smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if return_sequence:
            # Full token-level representation projected to out_dim (zero-out padding tokens)
            proj_seq = self.proj(outputs.last_hidden_state)
            return proj_seq * attention_mask.unsqueeze(-1).float()

        if self.pooling == "cls":
            # Extract [CLS] / <s> token representation
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)

        return self.proj(pooled)

    def extract_features(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Extract drug features from batch dict (matching GraphDTAModel interface).

        Reads `batch["drug_smiles_str"]` or fallbacks to SMILES strings.
        """
        device = next(self.parameters()).device

        if "drug_smiles_str" in batch:
            smiles_list = batch["drug_smiles_str"]
        elif "smiles" in batch:
            smiles_list = batch["smiles"]
        else:
            raise KeyError(
                "Batch does not contain 'drug_smiles_str' or 'smiles' for ChemBERTaEncoder."
            )

        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        return self.encode_smiles(smiles_list, device=device, return_sequence=return_sequence)

    def forward(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Standard forward pass delegating to extract_features."""
        return self.extract_features(batch, return_sequence=return_sequence)


@MODELS.register("esm2_encoder")
class ESM2Encoder(nn.Module):
    """Pretrained ESM-2 protein encoder for amino acid sequences.

    Architecture:
        AA seq -> ESM-2 Tokenizer -> ESM Transformer Body (320 or 480-dim)
               -> Masked Mean Pool -> Linear Projection -> out_dim (256-dim)

    Supported Base Models:
        - "facebook/esm2_t12_35M_UR50D" (12 layers, 480-dim, recommended)
        - "facebook/esm2_t6_8M_UR50D"   (6 layers, 320-dim, lightweight)

    Args:
        config: Dict containing:
            model_name            : HuggingFace model name (default: "facebook/esm2_t12_35M_UR50D")
            hidden_dim            : Inferred automatically from model if not given
            out_dim               : Projected output dim (default: 256)
            max_length            : Sequence length truncation threshold (default: 1024)
            pooling               : Pooling mode, "mean" or "bos" (default: "mean")
            freeze_backbone       : Freeze ESM-2 transformer layers (default: True)
            gradient_checkpointing: Enable gradient checkpointing to save VRAM (default: False)
            dropout               : Dropout rate after projection (default: 0.1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}
        self.model_name = cfg.get("model_name", "facebook/esm2_t12_35M_UR50D")
        self.out_dim = cfg.get("out_dim", 256)
        self.max_length = cfg.get("max_length", 1024)
        self.pooling = cfg.get("pooling", "mean")
        self.freeze_backbone = cfg.get("freeze_backbone", True)
        self.partially_unfrozen = False
        self.unfrozen_layers = 0
        self.use_grad_ckpt = cfg.get("gradient_checkpointing", False)
        self.return_sequence = cfg.get("return_sequence", False)
        dropout = cfg.get("dropout", 0.1)

        # Load HuggingFace tokenizer and ESM model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        if self.use_grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.hidden_dim = getattr(self.model.config, "hidden_size", 480)

        # Projection layer to unify representation dimension
        if self.out_dim != self.hidden_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj = nn.Identity()

        if self.freeze_backbone:
            self.freeze()

        # In-memory embedding cache for frozen backbone acceleration (saves 30x compute on DTI)
        self._embedding_cache: Dict[str, torch.Tensor] = {}

        # Fallback AA tokenizer for decoding token IDs if raw str is missing
        self._aa_tokenizer_fallback = AminoAcidTokenizer(max_length=self.max_length)

    def freeze(self) -> None:
        """Freeze all ESM transformer parameters for efficient feature extraction."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.freeze_backbone = True
        self.partially_unfrozen = False
        self.unfrozen_layers = 0

    def unfreeze(self, last_n_layers: Optional[int] = None) -> None:
        """Unfreeze transformer parameters (optionally only the top N layers)."""
        if last_n_layers is None:
            for param in self.model.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            self.partially_unfrozen = False
            self.unfrozen_layers = 0
        else:
            self.freeze()
            encoder_layers = getattr(getattr(self.model, "encoder", None), "layer", None)
            if encoder_layers is not None:
                for layer in encoder_layers[-last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            self.freeze_backbone = False
            self.partially_unfrozen = True
            self.unfrozen_layers = last_n_layers

    def _mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked mean pooling over non-padding AA residues."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_sequences(
        self,
        seq_list: List[str],
        device: torch.device,
        return_sequence: Optional[bool] = None,
    ) -> torch.Tensor:
        """Tokenize and encode amino acid sequences using ESM-2.

        Args:
            seq_list: List of amino acid sequence strings.
            device: Target torch device.
            return_sequence: If True, returns full token representation [B, L, out_dim].

        Returns:
            FloatTensor [B, out_dim] or [B, L, out_dim]
        """
        if return_sequence is None:
            return_sequence = self.return_sequence

        clean_seqs = [s if (s and isinstance(s, str)) else "A" for s in seq_list]

        # Branch 1: Partially unfrozen backbone (reuse cached frozen lower layers)
        if (
            self.partially_unfrozen
            and self.unfrozen_layers > 0
            and hasattr(self.model, "encoder")
            and getattr(self.model.encoder, "layer", None) is not None
            and len(self.model.encoder.layer) > self.unfrozen_layers
        ):
            encoder_layers = self.model.encoder.layer
            num_frozen = len(encoder_layers) - self.unfrozen_layers
            missing_indices = []
            missing_seqs = []
            seq_tensors: List[Optional[torch.Tensor]] = [None] * len(clean_seqs)

            for idx, seq in enumerate(clean_seqs):
                cache_key = f"inter_{num_frozen}_{seq}"
                if cache_key in self._embedding_cache:
                    seq_tensors[idx] = self._embedding_cache[cache_key].to(device)
                else:
                    missing_indices.append(idx)
                    missing_seqs.append(seq)

            if missing_seqs:
                encoded = self.tokenizer(
                    missing_seqs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.no_grad():
                    ext_mask = self.model.get_extended_attention_mask(
                        attention_mask, input_ids.shape
                    )
                    h = self.model.embeddings(input_ids=input_ids, attention_mask=attention_mask)
                    for layer in encoder_layers[:num_frozen]:
                        h = layer(h, attention_mask=ext_mask)[0]

                for i, orig_idx in enumerate(missing_indices):
                    act_len = int(attention_mask[i].sum().item())
                    s_vec = h[i, :act_len, :].detach().cpu().half()
                    self._embedding_cache[f"inter_{num_frozen}_{clean_seqs[orig_idx]}"] = s_vec
                    seq_tensors[orig_idx] = s_vec.to(device).float()

            valid_seqs = [
                s.float() if s.dtype != torch.float32 else s
                for s in seq_tensors
                if s is not None
            ]
            padded_h = torch.nn.utils.rnn.pad_sequence(valid_seqs, batch_first=True)
            lengths = [s.shape[0] for s in valid_seqs]
            max_len = padded_h.shape[1]
            attn_mask = (
                torch.arange(max_len, device=device)[None, :]
                < torch.tensor(lengths, device=device)[:, None]
            ).long()
            ext_mask = self.model.get_extended_attention_mask(
                attn_mask, padded_h.shape[:2]
            )

            # Pass through unfrozen upper layers with active gradients
            for layer in encoder_layers[-self.unfrozen_layers:]:
                padded_h = layer(padded_h, attention_mask=ext_mask)[0]

            if getattr(self.model.encoder, "emb_layer_norm_after", None) is not None:
                padded_h = self.model.encoder.emb_layer_norm_after(padded_h)

            if return_sequence:
                proj_seq = self.proj(padded_h)
                pad_mask = (padded_h.abs().sum(dim=-1, keepdim=True) > 0).float()
                return proj_seq * pad_mask
            else:
                if self.pooling == "bos":
                    pooled = padded_h[:, 0, :]
                else:
                    pooled = self._mean_pooling(padded_h, attn_mask)
                return self.proj(pooled)

        if return_sequence:
            if self.freeze_backbone:
                missing_indices = []
                missing_seqs = []
                seq_tensors: List[Optional[torch.Tensor]] = [None] * len(clean_seqs)

                for idx, seq in enumerate(clean_seqs):
                    cache_key = f"seq_{seq}"
                    if cache_key in self._embedding_cache:
                        seq_tensors[idx] = self._embedding_cache[cache_key].to(device)
                    else:
                        missing_indices.append(idx)
                        missing_seqs.append(seq)

                if missing_seqs:
                    encoded = self.tokenizer(
                        missing_seqs,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        raw_hidden = outputs.last_hidden_state

                    for i, orig_idx in enumerate(missing_indices):
                        act_len = int(attention_mask[i].sum().item())
                        # Store unpadded raw hidden state tokens on CPU in half precision
                        s_vec = raw_hidden[i, :act_len, :].detach().cpu().half()
                        self._embedding_cache[f"seq_{clean_seqs[orig_idx]}"] = s_vec
                        seq_tensors[orig_idx] = s_vec.to(device).float()

                # Pad sequences across the batch dimension and apply projection
                valid_seqs = [
                    s.float() if s.dtype != torch.float32 else s
                    for s in seq_tensors
                    if s is not None
                ]
                padded = torch.nn.utils.rnn.pad_sequence(valid_seqs, batch_first=True)
                proj_seq = self.proj(padded)
                # Keep padding positions strictly zeroed out
                pad_mask = (padded.abs().sum(dim=-1, keepdim=True) > 0).float()
                return proj_seq * pad_mask
            else:
                encoded = self.tokenizer(
                    clean_seqs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                proj_seq = self.proj(outputs.last_hidden_state)
                return proj_seq * attention_mask.unsqueeze(-1).float()

        # Pooled vector extraction [B, out_dim]
        if self.freeze_backbone:
            missing_indices = []
            missing_seqs = []
            pooled_list: List[Optional[torch.Tensor]] = [None] * len(clean_seqs)

            for idx, seq in enumerate(clean_seqs):
                cache_key = f"pool_{seq}"
                if cache_key in self._embedding_cache:
                    pooled_list[idx] = self._embedding_cache[cache_key].to(device)
                elif seq in self._embedding_cache:
                    pooled_list[idx] = self._embedding_cache[seq].to(device)
                else:
                    missing_indices.append(idx)
                    missing_seqs.append(seq)

            if missing_seqs:
                encoded = self.tokenizer(
                    missing_seqs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if self.pooling == "bos":
                        missing_pooled = outputs.last_hidden_state[:, 0, :]
                    else:
                        missing_pooled = self._mean_pooling(
                            outputs.last_hidden_state, attention_mask
                        )

                for i, orig_idx in enumerate(missing_indices):
                    p_vec = missing_pooled[i].detach()
                    # Cache representation on CPU to avoid GPU VRAM buildup
                    self._embedding_cache[f"pool_{clean_seqs[orig_idx]}"] = p_vec.cpu()
                    pooled_list[orig_idx] = p_vec

            pooled = torch.stack([p for p in pooled_list if p is not None], dim=0)
        else:
            encoded = self.tokenizer(
                clean_seqs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            if self.pooling == "bos":
                pooled = outputs.last_hidden_state[:, 0, :]
            else:
                pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)

        return self.proj(pooled)

    def encode_sequence(
        self, seq_input: Any, return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Encode sequence input, maintaining backward compatibility with ProteinCNNEncoder.

        Supports:
        - List of strings: `["MKTAY...", "MSHHW..."]`
        - LongTensor [B, max_len]: Decoded via AminoAcidTokenizer before ESM tokenization.
        - Dict (Batch): Reads `target_seq_str` or `target_seq`.
        """
        device = next(self.parameters()).device

        if isinstance(seq_input, dict):
            return self.extract_features(seq_input, return_sequence=return_sequence)

        if isinstance(seq_input, (list, tuple)):
            return self.encode_sequences(
                list(seq_input), device=device, return_sequence=return_sequence
            )

        if isinstance(seq_input, torch.Tensor):
            # Decode tokenized tensor back to AA string list
            seq_list = [self._aa_tokenizer_fallback.decode(row) for row in seq_input]
            return self.encode_sequences(seq_list, device=device, return_sequence=return_sequence)

        if isinstance(seq_input, str):
            return self.encode_sequences(
                [seq_input], device=device, return_sequence=return_sequence
            )

        raise TypeError(f"Unsupported sequence input type for ESM2Encoder: {type(seq_input)}")

    def extract_features(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Extract protein features from batch dict."""
        device = next(self.parameters()).device

        if "target_seq_str" in batch:
            seq_list = batch["target_seq_str"]
            if isinstance(seq_list, str):
                seq_list = [seq_list]
            return self.encode_sequences(seq_list, device=device, return_sequence=return_sequence)

        if "target_seq" in batch:
            return self.encode_sequence(batch["target_seq"], return_sequence=return_sequence)

        raise KeyError("Batch does not contain 'target_seq_str' or 'target_seq' for ESM2Encoder.")

    def forward(
        self, batch: Dict[str, Any], return_sequence: Optional[bool] = None
    ) -> torch.Tensor:
        """Standard forward pass."""
        return self.extract_features(batch, return_sequence=return_sequence)
