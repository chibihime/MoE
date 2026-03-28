from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict
import numpy as np

from embedding_layer import MultimodalEmbeddingLayer
from moe_fusion_layer import MoEFusionLayer


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
# Model
# =========================

class DEAPEmotionModel(nn.Module):

    DIMENSIONS = ("valence", "arousal", "dominance", "liking")

    def __init__(
        self,
        physio_input_dim=32,
        audio_input_dim=3,
        embed_dim=64,
        physio_hidden_dim=128,
        audio_hidden_dim=32,
        expert_hidden_dim=128,
        gate_hidden_dim=64,
        dropout=0.1,
        num_modalities=2,
        loss_weights=None,
        lb_lambda=0.01,
        ccc_lambda=0.05,   # ✅ REDUCED (important)
    ):
        super().__init__()

        self.embedding_layer = MultimodalEmbeddingLayer(
            physio_input_dim,
            audio_input_dim,
            embed_dim,
            physio_hidden_dim,
            audio_hidden_dim,
            dropout,
            num_modalities,
        )

        self.fusion_layer = MoEFusionLayer(
            embed_dim,
            expert_hidden_dim,
            gate_hidden_dim,
            dropout,
        )

        self.loss_weights = {d: 1.0 for d in self.DIMENSIONS}
        if loss_weights:
            self.loss_weights.update(loss_weights)

        self.lb_lambda = lb_lambda
        self.ccc_lambda = ccc_lambda

    # =========================
    # Forward
    # =========================

    def forward(self, physio_feats, audio_feats, modality_mask):
        embeddings = self.embedding_layer(physio_feats, audio_feats, modality_mask)
        return self.fusion_layer(embeddings)

    # =========================
    # Training
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

        # ✅ Encourage variance (stronger)
        all_preds = torch.cat([predictions[d] for d in self.DIMENSIONS], dim=1)
        var_loss = -torch.mean(torch.std(all_preds, dim=0))

        total_loss = (
            task_loss
            + self.ccc_lambda * ccc_loss
            + self.lb_lambda * lb_loss
            + 0.05 * var_loss   # ✅ increased
        )

        print("\n[DEBUG]")
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
    # Predict (FIXED)
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
            pred = torch.tanh(predictions[dim] * 1.5)  # ✅ expand range
            result[dim] = pred.cpu().numpy()

        if return_gate_weights:
            result["gate_weights"] = predictions["gate_weights"].cpu()

        return result

    # =========================
    # Optimizer
    # =========================

    def configure_optimizer(self, lr=3e-4, weight_decay=1e-4, t_max=50):
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

    # =========================
    # Save / Load
    # =========================

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict()}, path)
        print(f"[Model] Saved to {path}")

    @classmethod
    def load(cls, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"[Model] Loaded from {path}")
        return model