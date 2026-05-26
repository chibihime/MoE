"""
models/text/encoder.py

Japanese text encoder using cl-tohoku/bert-base-japanese-v3.
Outputs a fixed-size feature vector per input for use by the MoE router and fusion layer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict


class JapaneseTextEncoder(nn.Module):
    """
    Encodes Japanese text input into dense feature vectors.

    - Backbone: cl-tohoku/bert-base-japanese-v3 (12 layers, 768-dim)
    - First `freeze_layers` transformer layers are frozen to preserve
      pre-trained Japanese language representations.
    - [CLS] token output is projected to a common feature space.
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_name = config.get("model_name", "cl-tohoku/bert-base-japanese-v3")
        self.hidden_size = config.get("hidden_size", 768)
        freeze_layers = config.get("freeze_layers", 8)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self._freeze_layers(freeze_layers)

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _freeze_layers(self, num_layers: int):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.encoder.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) optional

        Returns dict with keys:
            features:        (batch, hidden_size)  — projected CLS, used by router/fusion
            sequence_output: (batch, seq, hidden)  — full token outputs, for future attention use
            pooled:          (batch, hidden_size)  — raw CLS before projection
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        return {
            "features": self.projection(cls),
            "sequence_output": outputs.last_hidden_state,
            "pooled": cls,
        }

    def tokenize(
        self,
        texts: list,
        max_length: int = 512,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a list of Japanese strings and move tensors to device."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)