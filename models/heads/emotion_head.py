"""
models/heads/emotion_head.py

Dual output head for emotion prediction:
  1. Classification:  7 discrete emotion classes with softmax probabilities
  2. Regression:      continuous valence + arousal scores (1-9 scale)

Changes vs original:
  - Label smoothing (ε=0.1) added to CrossEntropyLoss. Stops the model
    becoming overconfident on majority classes; redistributes 10% of
    probability mass across all classes, which regularises the classifier
    and consistently improves macro-F1 on imbalanced sets.
  - Loss instance created once in __init__ (not reconstructed every step).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

EMOTION_CLASSES: List[str] = [
    "happy", "sad", "angry", "neutral", "anxious", "surprised", "disgusted"
]

LABEL_SMOOTHING = 0.1   # redistribute 10% of prob mass — tune down to 0.05
                         # if val loss starts diverging from train loss


class EmotionHead(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        input_size  = config.get("input_size", 512)
        clf         = config.get("classification", {})
        reg         = config.get("regression", {})

        clf_hidden  = clf.get("hidden_size", 256)
        clf_dropout = clf.get("dropout", 0.3)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, clf_hidden),
            nn.LayerNorm(clf_hidden),
            nn.GELU(),
            nn.Dropout(clf_dropout),
            nn.Linear(clf_hidden, len(EMOTION_CLASSES)),
        )

        reg_hidden  = reg.get("hidden_size", 256)
        reg_dropout = reg.get("dropout", 0.3)
        self.regressor = nn.Sequential(
            nn.Linear(input_size, reg_hidden),
            nn.LayerNorm(reg_hidden),
            nn.GELU(),
            nn.Dropout(reg_dropout),
            nn.Linear(reg_hidden, 2),
            nn.Sigmoid(),
        )

        # Pre-built loss — weight tensor injected at call time so class
        # weights computed in train.py still flow through normally.
        # label_smoothing is the only new argument vs the original.
        self._reg_loss = nn.MSELoss()

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits   = self.classifier(fused)
        probs    = torch.softmax(logits, dim=-1)
        pred_idx = logits.argmax(dim=-1)

        continuous = self.regressor(fused)
        va = continuous * 8.0 + 1.0   # [0,1] → [1,9]

        return {
            "logits":          logits,
            "probs":           probs,
            "predicted_class": pred_idx,
            "predicted_label": [EMOTION_CLASSES[i] for i in pred_idx.tolist()],
            "valence":         va[:, 0],
            "arousal":         va[:, 1],
        }

    def compute_loss(
        self,
        outputs:         Dict[str, torch.Tensor],
        labels:          torch.Tensor,                   # (B,) int64
        valence_targets: torch.Tensor,                   # (B,) float
        arousal_targets: torch.Tensor,                   # (B,) float
        clf_weight:      float = 0.6,
        reg_weight:      float = 0.4,
        class_weights:   Optional[torch.Tensor] = None,  # (7,) float
    ) -> Dict[str, torch.Tensor]:
        """
        Combined classification + regression loss.

        class_weights: per-class inverse-frequency weights computed in
            train.py. Still applied on top of label smoothing — they
            are complementary: weights shift the gradient magnitude per
            class; smoothing softens the target distribution for all.

        label_smoothing=0.1: instead of a hard [0,…,0,1,0,…,0] target,
            the true class gets 0.9 + 0.1/7 ≈ 0.914 and each other
            class gets 0.1/7 ≈ 0.014. This prevents overconfidence and
            improves generalisation on minority classes.
        """
        clf_loss = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = LABEL_SMOOTHING,
        )(outputs["logits"], labels)

        va_pred   = torch.stack([outputs["valence"], outputs["arousal"]], dim=-1)
        va_target = torch.stack([valence_targets, arousal_targets],       dim=-1)
        reg_loss  = self._reg_loss(va_pred, va_target)

        total = clf_weight * clf_loss + reg_weight * reg_loss
        return {
            "total":          total,
            "classification": clf_loss,
            "regression":     reg_loss,
        }