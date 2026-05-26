"""
models/multimodal/router.py

Learned MoE router.

Decides which expert(s) to activate and their relative contribution weights,
based on the available modality features. Unlike a hard if/else switch, this
uses a small learned network so routing decisions adapt during training.

Experts: ["text", "audio"]  (indices 0, 1)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


EXPERTS = ["text", "audio"]


class MoERouter(nn.Module):
    """
    Computes soft routing weights over the two experts.

    - Both available:  gate network decides weights from combined features
    - One missing:     zero-fill the missing expert; renormalize weights
    - Temperature:     controls sharpness (low → near hard routing; default 1.0)
    """

    def __init__(self, config: Dict):
        super().__init__()
        hidden      = config.get("hidden_size", 256)
        input_size  = config.get("expert_hidden_size", 768)
        self.temp   = config.get("temperature", 1.0)

        # Input: concatenation of text + audio features (zero-filled if absent)
        self.router_net = nn.Sequential(
            nn.Linear(input_size * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, len(EXPERTS)),
        )

    def forward(
        self,
        text_features:  Optional[torch.Tensor] = None,  # (batch, 768)
        audio_features: Optional[torch.Tensor] = None,  # (batch, 768)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            weights:        (batch, 2)  — normalized expert weights
            text_weight:    (batch,)    — weight for text expert
            audio_weight:   (batch,)    — weight for audio expert
            routing_logits: (batch, 2)  — pre-softmax logits (for aux loss / logging)
        """
        ref   = text_features if text_features is not None else audio_features
        B, D, device = ref.shape[0], ref.shape[-1], ref.device
        zero  = torch.zeros(B, D, device=device)

        t = text_features  if text_features  is not None else zero
        a = audio_features if audio_features is not None else zero

        logits  = self.router_net(torch.cat([t, a], dim=-1)) / self.temp  # (B, 2)
        weights = torch.softmax(logits, dim=-1)

        # Mask unavailable experts and renormalize
        mask = torch.ones(B, 2, device=device)
        if text_features  is None: mask[:, 0] = 0.0
        if audio_features is None: mask[:, 1] = 0.0

        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return {
            "weights":        weights,
            "text_weight":    weights[:, 0],
            "audio_weight":   weights[:, 1],
            "routing_logits": logits,
        }