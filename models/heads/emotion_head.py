"""
models/heads/emotion_head.py

Dual output head for emotion prediction:
  1. Classification:  7 discrete emotion classes with softmax probabilities
  2. Regression:      continuous valence + arousal scores (1-9 scale)

Fix 1: compute_loss() now accepts class_weights tensor to counter
       class imbalance (happy/sad dominating, angry/anxious ignored).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

EMOTION_CLASSES: List[str] = [
    "happy", "sad", "angry", "neutral", "anxious", "surprised", "disgusted"
]


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
        labels:          torch.Tensor,            # (B,) int64
        valence_targets: torch.Tensor,            # (B,) float
        arousal_targets: torch.Tensor,            # (B,) float
        clf_weight:      float = 0.6,
        reg_weight:      float = 0.4,
        class_weights:   Optional[torch.Tensor] = None,  # (7,) float — Fix 1
    ) -> Dict[str, torch.Tensor]:
        """
        Combined classification + regression loss.

        class_weights: per-class weights for CrossEntropyLoss.
            Pass higher weights for under-represented classes
            (angry, anxious, surprised, disgusted) to stop the model
            collapsing to happy/sad. Computed automatically in train.py.
        """
        clf_loss = nn.CrossEntropyLoss(weight=class_weights)(
            outputs["logits"], labels
        )

        va_pred   = torch.stack([outputs["valence"], outputs["arousal"]], dim=-1)
        va_target = torch.stack([valence_targets, arousal_targets],       dim=-1)
        reg_loss  = nn.MSELoss()(va_pred, va_target)

        total = clf_weight * clf_loss + reg_weight * reg_loss
        return {
            "total":          total,
            "classification": clf_loss,
            "regression":     reg_loss,
        }