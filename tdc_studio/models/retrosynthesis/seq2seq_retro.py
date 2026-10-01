"""Neural Sequence-to-Sequence Retrosynthesis model with Beam Search."""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from rdkit import Chem

from tdc_studio.core.registry import MODELS
from tdc_studio.data.retrosyn import ReactionTokenizer
from tdc_studio.models.retrosynthesis.base import BaseRetroModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[:, : x.size(1)]


@MODELS.register("seq2seq_retro_model")
class Seq2SeqRetroModel(BaseRetroModel):
    """Transformer Encoder-Decoder for Sequence-to-Sequence chemical retrosynthesis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.d_model = int(self.config.get("d_model", 256))
        self.nhead = int(self.config.get("nhead", 8))
        self.num_encoder_layers = int(self.config.get("num_encoder_layers", 4))
        self.num_decoder_layers = int(self.config.get("num_decoder_layers", 4))
        self.dim_feedforward = int(self.config.get("dim_feedforward", 512))
        self.dropout = float(self.config.get("dropout", 0.1))
        self.max_length = int(self.config.get("max_length", 256))

        self.tokenizer = ReactionTokenizer(max_length=self.max_length)
        self.vocab_size = self.tokenizer.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.tokenizer.pad_token_id)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.max_length)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(self.d_model, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute training logits and sequence cross-entropy loss."""
        src = batch["input_ids"]  # [B, S]
        tgt = batch["target_ids"]  # [B, T]

        src_key_padding_mask = (src == self.tokenizer.pad_token_id)
        tgt_key_padding_mask = (tgt == self.tokenizer.pad_token_id)

        # Shift target for teacher forcing: decoder input is tgt[:, :-1], target is tgt[:, 1:]
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        tgt_pad_mask = tgt_key_padding_mask[:, :-1]

        t_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t_len).to(src.device)

        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt_input))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.fc_out(out)  # [B, T-1, V]
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_expected.reshape(-1))

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def predict_reactants(
        self,
        product_smiles: str,
        top_k: int = 5,
        reaction_type: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Generate top-k reactant candidate sequences using greedy/beam search."""
        self.eval()
        device = next(self.parameters()).device

        input_ids, attention_mask = self.tokenizer.encode(
            product_smiles,
            reaction_type=reaction_type,
            max_length=self.max_length,
        )
        src = input_ids.unsqueeze(0).to(device)  # [1, S]
        src_mask = (src == self.tokenizer.pad_token_id)
        src_emb = self.pos_encoder(self.embedding(src))

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

        # Simple multi-branch beam search (beam_width = top_k)
        beams = [([self.tokenizer.cls_token_id], 0.0)]
        completed_beams = []

        max_decode_len = min(128, self.max_length)

        for step in range(max_decode_len):
            new_candidates = []
            for seq, log_prob in beams:
                if seq[-1] == self.tokenizer.sep_token_id:
                    completed_beams.append((seq, log_prob))
                    continue

                tgt_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_emb = self.pos_encoder(self.embedding(tgt_tensor))
                t_len = tgt_tensor.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(t_len).to(device)

                out = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                )
                logits = self.fc_out(out[:, -1, :])  # [1, V]
                log_probs = torch.log_softmax(logits, dim=-1)

                top_log_probs, top_indices = torch.topk(log_probs, k=min(top_k * 2, self.vocab_size))
                for lp, idx in zip(top_log_probs[0], top_indices[0]):
                    new_candidates.append((seq + [idx.item()], log_prob + lp.item()))

            if not new_candidates:
                break

            # Keep top candidates
            new_candidates.sort(key=lambda x: x[1] / max(1, len(x[0])), reverse=True)
            beams = new_candidates[:top_k]

            if len(completed_beams) >= top_k:
                break

        completed_beams.extend(beams)
        completed_beams.sort(key=lambda x: x[1] / max(1, len(x[0])), reverse=True)

        results = []
        seen = set()
        for seq, score in completed_beams:
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            if not decoded or decoded in seen:
                continue

            # Validate chemical plausibility of predicted reactants
            parts = decoded.split(".")
            all_valid = True
            canonical_parts = []
            for part in parts:
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    all_valid = False
                    break
                canonical_parts.append(Chem.MolToSmiles(mol, canonical=True))

            if all_valid and canonical_parts:
                canon_reactants = ".".join(sorted(canonical_parts))
                if canon_reactants not in seen:
                    seen.add(canon_reactants)
                    norm_score = math.exp(score / max(1, len(seq)))
                    results.append((canon_reactants, round(norm_score, 4)))

            if len(results) >= top_k:
                break

        return results
