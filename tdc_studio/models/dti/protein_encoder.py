"""ProteinCNNEncoder — lightweight 1D-CNN encoder for amino acid sequences.

Design principles:
- Three parallel Conv1D kernels (sizes 4, 8, 12) capture local AA motifs at
  multiple granularities (secondary structure elements, binding motifs).
- Global max-pooling over each kernel → fixed-length vector regardless of seq length.
- Interface is intentionally identical to the future ESM2Encoder (Phase B):
  both expose encode_sequence(seq_tensor) -> Tensor, so GraphDTAModel can
  swap encoders via YAML config without any model code changes.
- Registered under "protein_cnn" in MODELS registry for YAML-driven config.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdc_studio.core.registry import MODELS
from tdc_studio.data.transforms import AminoAcidTokenizer


@MODELS.register("protein_cnn")
class ProteinCNNEncoder(nn.Module):
    """Multi-scale 1D-CNN encoder for protein amino acid sequences.

    Architecture (GraphDTA target-branch style):
        Embedding(25, embed_dim) → [Conv1D(k=4), Conv1D(k=8), Conv1D(k=12)]
                                 → GlobalMaxPool × 3 → Concat → FC → out_dim

    Args:
        config: Dict with keys:
            embed_dim   : AA embedding dimension (default 128).
            num_filters : Number of output channels per Conv1D (default 256).
            out_dim     : Final encoder output dimension (default 256).
                          Must match drug_encoder out_dim for BilinearFusion.
            dropout     : Dropout rate after FC (default 0.1).
    """

    VOCAB_SIZE = AminoAcidTokenizer.VOCAB_SIZE  # 25

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        embed_dim   = config.get("embed_dim",   128)
        num_filters = config.get("num_filters", 256)
        out_dim     = config.get("out_dim",     256)
        dropout     = config.get("dropout",     0.1)

        self.out_dim = out_dim

        # Embedding: maps AA token IDs → dense vectors (padding_idx=0 → zero grad)
        self.embedding = nn.Embedding(
            self.VOCAB_SIZE, embed_dim, padding_idx=0
        )

        # Three parallel 1D convolutions at different receptive fields:
        #   k=4  → dipeptide-level patterns (e.g. turn motifs)
        #   k=8  → short helical repeats
        #   k=12 → beta-strand / longer motifs
        self.conv4  = nn.Conv1d(embed_dim, num_filters, kernel_size=4,  padding=0)
        self.conv8  = nn.Conv1d(embed_dim, num_filters, kernel_size=8,  padding=0)
        self.conv12 = nn.Conv1d(embed_dim, num_filters, kernel_size=12, padding=0)

        # Project concatenated multi-scale features → unified protein representation
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 3, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def encode_sequence(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of padded AA integer sequences.

        Args:
            seq_tensor: LongTensor of shape [B, max_len] from AminoAcidTokenizer.

        Returns:
            FloatTensor of shape [B, out_dim] — protein representation.
        """
        # [B, L] → [B, L, embed_dim] → [B, embed_dim, L]  (Conv1d expects channels first)
        x = self.embedding(seq_tensor).permute(0, 2, 1)

        # Convolve + global max-pool: each → [B, num_filters]
        def _conv_pool(conv: nn.Conv1d, h: torch.Tensor) -> torch.Tensor:
            out = F.relu(conv(h))                            # [B, num_filters, L']
            return F.max_pool1d(out, out.size(2)).squeeze(2) # [B, num_filters]

        h4  = _conv_pool(self.conv4,  x)
        h8  = _conv_pool(self.conv8,  x)
        h12 = _conv_pool(self.conv12, x)

        # Concatenate all scales → [B, num_filters * 3]
        h_cat = torch.cat([h4, h8, h12], dim=1)
        return self.fc(h_cat)  # [B, out_dim]

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Standard forward pass reading from batch dict.

        Reads batch["target_seq"] (LongTensor [B, max_len]).
        Delegates to encode_sequence() so subclasses can override easily.
        """
        return self.encode_sequence(batch["target_seq"])
