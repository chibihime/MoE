"""
models/fusion/fusion.py

Fuses text and audio expert features using a learned gating mechanism.

The gate learns to weight each modality's contribution dynamically,
rather than using a fixed 50/50 or rule-based split. This means the
model can learn to trust audio more when text is sparse, and vice versa.

Also handles graceful degradation: if only one modality is available
(e.g. audio-only stream, or text-only input), it falls back cleanly
without any special-casing in the calling code.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class WeightedFusion(nn.Module):
    """
    Fuses text and audio features into a single vector for the emotion head.

    Modes:
      Both modalities:   learned gate weights text_proj and audio_proj
      Text only:         passes text_proj directly (no gate needed)
      Audio only:        passes audio_proj directly
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_size     = config.get("hidden_size", 512)
        expert_hidden        = config.get("expert_hidden_size", 768)

        # Project each expert from its hidden size into the shared fusion space
        self.text_proj  = nn.Linear(expert_hidden, self.hidden_size)
        self.audio_proj = nn.Linear(expert_hidden, self.hidden_size)

        # Learned gate: takes both projections concatenated → 2 scalar weights
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

        self.norm    = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

    def forward(
        self,
        text_features:   Optional[torch.Tensor] = None,  # (batch, 768)
        audio_features:  Optional[torch.Tensor] = None,  # (batch, 768)
        text_available:  bool = True,
        audio_available: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            text_features:   encoder output for text expert, or None
            audio_features:  encoder output for audio expert, or None
            text_available:  whether text input was provided this call
            audio_available: whether audio input was provided this call

        Returns:
            fused: (batch, hidden_size) — ready for emotion head
        """
        ref = text_features if text_features is not None else audio_features
        batch, device = ref.shape[0], ref.device
        zero = torch.zeros(batch, self.hidden_size, device=device)

        t_proj = self.text_proj(text_features)   if (text_available  and text_features  is not None) else zero
        a_proj = self.audio_proj(audio_features) if (audio_available and audio_features is not None) else zero

        if text_available and audio_available:
            gate_w = self.gate(torch.cat([t_proj, a_proj], dim=-1))  # (batch, 2)
            fused  = gate_w[:, 0:1] * t_proj + gate_w[:, 1:2] * a_proj
        elif text_available:
            fused = t_proj
        else:
            fused = a_proj

        return self.norm(self.dropout(fused))