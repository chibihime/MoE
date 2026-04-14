from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np

from embedding_layer import MultimodalEmbeddingLayer
from moe_fusion_layer import MoEFusionLayer


# =========================
# Type alias
# =========================

SequenceModelType = Literal["lstm", "rnn", "transformer"]


# =========================
# Dataclasses
# =========================

@dataclass
class TrainStepOutput:
    loss: torch.Tensor
    task_loss: torch.Tensor
    lb_loss: torch.Tensor
    per_dim_losses: Dict[str, torch.Tensor]
    gate_weights: torch.Tensor
    predictions: Dict[str, torch.Tensor]


@dataclass
class ValStepOutput:
    rmse: Dict[str, float]
    ccc: Dict[str, float]
    mean_rmse: float
    mean_ccc: float


# =========================
# CCC
# =========================

def concordance_correlation_coefficient(preds, targets, eps=1e-8):
    preds = preds.view(-1)
    targets = targets.view(-1)

    mu_p = preds.mean()
    mu_t = targets.mean()

    var_p = preds.var(unbiased=False)
    var_t = targets.var(unbiased=False)

    cov = ((preds - mu_p) * (targets - mu_t)).mean()

    return (2 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + eps)

# =========================
# Forward (UPDATED)
# =========================

def forward(self, physio_feats, audio_feats, modality_mask):

    # Step 1: Embedding
    embeddings = self.embedding_layer(physio_feats, audio_feats, modality_mask)

    physio = embeddings["physio"]
    audio = embeddings["audio"]
    context = embeddings["context"]

    # Step 2: MoE Fusion
    fused_out = self.fusion_layer(embeddings)
    fused = fused_out["fused"]                 # (B, D)
    gate_weights = fused_out["gate_weights"]   # (B, num_experts)

    # Step 3: Build REAL sequence (B, 4, D)
    sequence = torch.stack(
        [physio, audio, context, fused],
        dim=1
    )

    # Step 4: Sequence model
    seq_out = self.seq_model(sequence)

    # Step 5: Heads
    predictions = {dim: self.heads[dim](seq_out) for dim in self.DIMENSIONS}
    predictions["gate_weights"] = gate_weights

    return predictions

# =========================
# Sequence Models
# =========================

class LSTMSequenceModel(nn.Module):
    """Bi-directional LSTM. Input: (B, T, embed_dim) → Output: (B, embed_dim*2)"""

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, embed_dim)  or  (B, embed_dim) — handle both
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, embed_dim)
        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers*2, B, hidden)
        # Concat last forward & backward hidden states
        fwd = h_n[-2]                   # (B, hidden)
        bwd = h_n[-1]                   # (B, hidden)
        return torch.cat([fwd, bwd], dim=-1)   # (B, hidden*2)


class RNNSequenceModel(nn.Module):
    """Vanilla Bi-directional RNN. Input: (B, T, embed_dim) → Output: (B, embed_dim*2)"""

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, h_n = self.rnn(x)            # h_n: (num_layers*2, B, hidden)
        fwd = h_n[-2]
        bwd = h_n[-1]
        return torch.cat([fwd, bwd], dim=-1)


class TransformerSequenceModel(nn.Module):
    """
    Lightweight Transformer encoder.
    Input: (B, T, embed_dim) or (B, embed_dim) → Output: (B, embed_dim)
    Uses [CLS] token pooling.
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, embed_dim)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)              # (B, T+1, embed_dim)
        x = self.encoder(x)
        return x[:, 0]                               # CLS token → (B, embed_dim)


def build_sequence_model(
    model_type: SequenceModelType,
    embed_dim: int,
    seq_hidden_dim: int,
    seq_num_layers: int,
    dropout: float,
) -> nn.Module:
    """Factory that returns the right sequence model and its output dim."""
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMSequenceModel(embed_dim, seq_hidden_dim, seq_num_layers, dropout)
    elif model_type == "rnn":
        return RNNSequenceModel(embed_dim, seq_hidden_dim, seq_num_layers, dropout)
    elif model_type == "transformer":
        nhead = max(1, embed_dim // 16)  # auto-pick heads; embed_dim must be divisible
        # Ensure embed_dim is divisible by nhead
        while embed_dim % nhead != 0 and nhead > 1:
            nhead -= 1
        return TransformerSequenceModel(
            embed_dim, nhead=nhead, num_layers=seq_num_layers,
            ff_dim=seq_hidden_dim * 2, dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown sequence_model_type='{model_type}'. "
            "Choose from: 'lstm', 'rnn', 'transformer'."
        )


# =========================
# Per-dimension prediction heads
# =========================

class DimensionHead(nn.Module):
    """Small MLP head for a single emotion dimension."""

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Model
# =========================

class DEAPEmotionModel(nn.Module):

    DIMENSIONS = ("valence", "arousal", "dominance", "liking")

    def __init__(
        self,
        physio_input_dim: int = 32,
        audio_input_dim: int = 3,
        embed_dim: int = 64,
        physio_hidden_dim: int = 128,
        audio_hidden_dim: int = 32,
        expert_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.1,
        num_modalities: int = 2,
        loss_weights: Dict[str, float] | None = None,
        lb_lambda: float = 0.01,
        ccc_lambda: float = 0.05,
        # ── Sequence model config ──────────────────────────────────────────
        sequence_model_type: SequenceModelType = "lstm",
        seq_hidden_dim: int = 128,   # LSTM/RNN hidden size; Transformer ff_dim = seq_hidden_dim*2
        seq_num_layers: int = 2,
    ):
        super().__init__()

        self.sequence_model_type = sequence_model_type.lower()

        # 1. Embedding layer  ──────────────────────────────────────────────
        self.embedding_layer = MultimodalEmbeddingLayer(
            physio_input_dim,
            audio_input_dim,
            embed_dim,
            physio_hidden_dim,
            audio_hidden_dim,
            dropout,
            num_modalities,
        )

        # 2. Fusion layer (MoE)  ───────────────────────────────────────────
        self.fusion_layer = MoEFusionLayer(
            embed_dim,
            expert_hidden_dim,
            gate_hidden_dim,
            dropout,
        )

        # 3. Sequence model  ───────────────────────────────────────────────
        self.seq_model = build_sequence_model(
            sequence_model_type,
            embed_dim=embed_dim,
            seq_hidden_dim=seq_hidden_dim,
            seq_num_layers=seq_num_layers,
            dropout=dropout,
        )
        seq_out_dim = self.seq_model.out_dim   # varies per model type

        # 4. Per-dimension prediction heads  ───────────────────────────────
        self.heads = nn.ModuleDict({
            dim: DimensionHead(seq_out_dim, dropout)
            for dim in self.DIMENSIONS
        })

        # 5. Loss config  ──────────────────────────────────────────────────
        self.loss_weights = {d: 1.0 for d in self.DIMENSIONS}
        if loss_weights:
            self.loss_weights.update(loss_weights)

        self.lb_lambda = lb_lambda
        self.ccc_lambda = ccc_lambda

    # =========================
    # Forward
    # =========================

    def forward(self, physio_feats, audio_feats, modality_mask):
        # Step 1: Embed
        embeddings = self.embedding_layer(physio_feats, audio_feats, modality_mask)

        # Step 2: MoE Fusion → dict with 'fused' tensor + 'gate_weights'
        fused_out = self.fusion_layer(embeddings)           # dict
        fused = fused_out["fused"]                          # (B, embed_dim)
        gate_weights = fused_out["gate_weights"]            # (B, num_experts)

        # Step 3: Sequence model
        seq_out = self.seq_model(fused)                     # (B, seq_out_dim)

        # Step 4: Dimension heads
        predictions = {dim: self.heads[dim](seq_out) for dim in self.DIMENSIONS}
        predictions["gate_weights"] = gate_weights

        return predictions

    # =========================
    # Training step
    # =========================

    def training_step(self, physio_feats, audio_feats, modality_mask, targets):

        predictions = self.forward(physio_feats, audio_feats, modality_mask)

        task_loss = 0.0
        ccc_loss = 0.0
        per_dim_losses = {}

        for dim in self.DIMENSIONS:
            pred = predictions[dim]
            target = targets[dim].to(pred.device)

            if target.dim() == 1:
                target = target.unsqueeze(-1)

            loss = nn.functional.smooth_l1_loss(pred, target)
            per_dim_losses[dim] = loss
            task_loss += self.loss_weights[dim] * loss

            ccc = concordance_correlation_coefficient(pred, target)
            ccc_loss += (1 - ccc)

        lb_loss = MoEFusionLayer.load_balance_loss(predictions["gate_weights"])

        # Encourage variance across dimensions
        all_preds = torch.cat([predictions[d] for d in self.DIMENSIONS], dim=1)
        var_loss = -torch.mean(torch.std(all_preds, dim=0))

        total_loss = (
            task_loss
            + self.ccc_lambda * ccc_loss
            + self.lb_lambda * lb_loss
            + 0.05 * var_loss
        )

        print(f"\n[DEBUG] seq_model={self.sequence_model_type}")
        print("pred mean:", all_preds.mean().item())
        print("pred std :", all_preds.std().item())
        print("task_loss:", task_loss.item())
        print("ccc_loss :", ccc_loss.item())
        print("lb_loss  :", lb_loss.item())

        

        return TrainStepOutput(
            loss=total_loss,
            task_loss=task_loss,
            lb_loss=lb_loss.detach(),
            per_dim_losses=per_dim_losses,
            gate_weights=predictions["gate_weights"].detach(),
            predictions={k: predictions[k].detach() for k in self.DIMENSIONS},
        )


    # =========================
    # Predict
    # =========================

    @torch.no_grad()
    def predict(
        self,
        physio_feats,
        audio_feats,
        modality_mask,
        return_gate_weights: bool = False,
    ):
        predictions = self.forward(physio_feats, audio_feats, modality_mask)

        result = {}
        for dim in self.DIMENSIONS:
            pred = torch.tanh(predictions[dim] * 1.5)   # expand range
            result[dim] = pred.cpu().numpy()

        if return_gate_weights:
            result["gate_weights"] = predictions["gate_weights"].cpu()

        return result

    # =========================
    # Optimizer
    # =========================

    def configure_optimizer(self, lr: float = 3e-4, weight_decay: float = 1e-4, t_max: int = 50):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
        )
        return optimizer, scheduler

    @torch.no_grad()
    def validation_step(self, physio_feats, audio_feats, modality_mask, targets):

        predictions = self.forward(physio_feats, audio_feats, modality_mask)

        rmse = {}
        ccc = {}

        for dim in self.DIMENSIONS:
            pred = predictions[dim]
            target = targets[dim].to(pred.device)

        if target.dim() == 1:
            target = target.unsqueeze(-1)

        mse = torch.mean((pred - target) ** 2)
        rmse[dim] = torch.sqrt(mse).item()

        ccc_val = concordance_correlation_coefficient(pred, target)
        ccc[dim] = ccc_val.item()

        mean_rmse = float(np.mean(list(rmse.values())))
        mean_ccc = float(np.mean(list(ccc.values())))

        return ValStepOutput(
        rmse=rmse,
        ccc=ccc,
        mean_rmse=mean_rmse,
        mean_ccc=mean_ccc,
        )
    # =========================
    # Save / Load
    # =========================

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "sequence_model_type": self.sequence_model_type,
        }, path)
        print(f"[Model] Saved to {path}")

    @classmethod
    def load(cls, path: str, map_location=None, **kwargs):
        checkpoint = torch.load(path, map_location=map_location)
        seq_type = checkpoint.get("sequence_model_type", "lstm")
        model = cls(sequence_model_type=seq_type, **kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"[Model] Loaded from {path} (seq_model={seq_type})")
        return model

    # =========================
    # Convenience: build all three for comparison
    # =========================

    @classmethod
    def build_all(cls, **shared_kwargs) -> Dict[str, "DEAPEmotionModel"]:
        """
        Instantiate LSTM, RNN, and Transformer variants with identical settings.

        Usage:
            models = DEAPEmotionModel.build_all(embed_dim=64, seq_hidden_dim=128)
            for name, model in models.items():
                optimizer, scheduler = model.configure_optimizer()
                # ... train loop ...
        """
        return {
            smt: cls(sequence_model_type=smt, **shared_kwargs)
            for smt in ("lstm", "rnn", "transformer")
        }
    
    