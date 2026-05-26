"""
models/heads/emotion_head.py

Dual output head for emotion prediction:
  1. Classification:  7 discrete emotion classes with softmax probabilities
  2. Regression:      continuous valence + arousal scores on DEAP 1–9 scale

Both heads share the same fused feature input from the fusion layer.
"""

import torch
import torch.nn as nn
from typing import Dict, List

EMOTION_CLASSES: List[str] = [
    "happy", "sad", "angry", "neutral", "anxious", "surprised", "disgusted"
]


class EmotionHead(nn.Module):
    """
    Takes fused multimodal features and produces:
      - logits + probabilities for 7 emotion classes
      - valence and arousal scores (1–9 scale)
    """

    def __init__(self, config: Dict):
        super().__init__()
        input_size = config.get("input_size", 512)

        clf = config.get("classification", {})
        reg = config.get("regression", {})

        # --- Classification head ---
        clf_hidden  = clf.get("hidden_size", 256)
        clf_dropout = clf.get("dropout", 0.3)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, clf_hidden),
            nn.LayerNorm(clf_hidden),
            nn.GELU(),
            nn.Dropout(clf_dropout),
            nn.Linear(clf_hidden, len(EMOTION_CLASSES)),
        )

        # --- Regression head (valence + arousal) ---
        reg_hidden  = reg.get("hidden_size", 256)
        reg_dropout = reg.get("dropout", 0.3)
        self.regressor = nn.Sequential(
            nn.Linear(input_size, reg_hidden),
            nn.LayerNorm(reg_hidden),
            nn.GELU(),
            nn.Dropout(reg_dropout),
            nn.Linear(reg_hidden, 2),   # [valence, arousal]
            nn.Sigmoid(),               # → [0, 1]; scaled to [1, 9] in forward
        )

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (batch, input_size) — output of fusion layer

        Returns dict with keys:
            logits:           (batch, 7)   — raw classification logits
            probs:            (batch, 7)   — softmax probabilities
            predicted_class:  (batch,)     — argmax class index
            predicted_label:  list[str]    — emotion label strings
            valence:          (batch,)     — score in [1, 9]
            arousal:          (batch,)     — score in [1, 9]
        """
        logits = self.classifier(fused)
        probs  = torch.softmax(logits, dim=-1)
        pred_idx = logits.argmax(dim=-1)

        continuous = self.regressor(fused)            # (batch, 2) in [0, 1]
        va = continuous * 8.0 + 1.0                   # scale to [1, 9]

        return {
            "logits":           logits,
            "probs":            probs,
            "predicted_class":  pred_idx,
            "predicted_label":  [EMOTION_CLASSES[i] for i in pred_idx.tolist()],
            "valence":          va[:, 0],
            "arousal":          va[:, 1],
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,            # (batch,) int64 class indices
        valence_targets: torch.Tensor,   # (batch,) float, 1–9
        arousal_targets: torch.Tensor,   # (batch,) float, 1–9
        clf_weight: float = 0.6,
        reg_weight: float = 0.4,
    ) -> Dict[str, torch.Tensor]:
        """
        Combined classification + regression loss.

        Args:
            outputs:          forward() output dict
            labels:           discrete emotion class indices
            valence_targets:  DEAP valence scores
            arousal_targets:  DEAP arousal scores
            clf_weight:       weight for classification loss term
            reg_weight:       weight for regression loss term

        Returns dict with keys: total, classification, regression
        """
        clf_loss = nn.CrossEntropyLoss()(outputs["logits"], labels)

        va_pred   = torch.stack([outputs["valence"], outputs["arousal"]], dim=-1)
        va_target = torch.stack([valence_targets, arousal_targets], dim=-1)
        reg_loss  = nn.MSELoss()(va_pred, va_target)

        total = clf_weight * clf_loss + reg_weight * reg_loss

        return {
            "total":          total,
            "classification": clf_loss,
            "regression":     reg_loss,
        }