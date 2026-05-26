import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

"""
models/multimodal/moe.py

MoEEmotionModel — top-level model wiring all components.

forward() accepts either:
  A) Raw inputs  → runs encoders internally (used at inference)
  B) Pre-computed features (text_features / audio_features kwargs)
     → bypasses encoders entirely (used during cached training)

Data flow:
  Text  → JapaneseTextEncoder → text_features  ─┐
                                                  ├─ MoERouter
  Audio → AudioEncoder        → audio_features ─┘
                                                  │
                                               WeightedFusion
                                                  │
                                               EmotionHead → predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from models.text.encoder    import JapaneseTextEncoder
from models.audio.encoder   import AudioEncoder
from models.fusion.fusion   import WeightedFusion
from models.heads.emotion_head import EmotionHead
from models.multimodal.router  import MoERouter


class MoEEmotionModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.text_encoder  = JapaneseTextEncoder(config["text_encoder"])
        self.audio_encoder = AudioEncoder(config["audio_encoder"])
        self.router        = MoERouter(config["router"])
        self.fusion        = WeightedFusion(config["fusion"])

        head_cfg = {**config["heads"], "input_size": config["fusion"]["hidden_size"]}
        self.head = EmotionHead(head_cfg)

    def forward(
        self,
        # Raw inputs (inference path)
        input_ids:            Optional[torch.Tensor] = None,
        attention_mask:       Optional[torch.Tensor] = None,
        token_type_ids:       Optional[torch.Tensor] = None,
        audio_values:         Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        # Pre-computed features (cached training path — bypasses encoders)
        text_features:        Optional[torch.Tensor] = None,
        audio_features:       Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        At least one modality must be provided (raw or pre-computed).
        Pre-computed features take priority over raw inputs when both are given.
        """
        # Resolve text features
        if text_features is None and input_ids is not None:
            out = self.text_encoder(input_ids, attention_mask, token_type_ids)
            text_features = out["features"]

        # Resolve audio features
        if audio_features is None and audio_values is not None:
            out = self.audio_encoder(audio_values, audio_attention_mask)
            audio_features = out["features"]

        if text_features is None and audio_features is None:
            raise ValueError("At least one modality must be provided.")

        text_avail  = text_features  is not None
        audio_avail = audio_features is not None

        routing = self.router(text_features, audio_features)
        fused   = self.fusion(
            text_features=text_features,
            audio_features=audio_features,
            text_available=text_avail,
            audio_available=audio_avail,
        )
        predictions = self.head(fused)
        predictions["routing"] = routing
        return predictions

    # ── Convenience inference methods ──────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        text:   Optional[str]        = None,
        audio:  Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> Dict:
        self.eval()
        input_ids = attention_mask = token_type_ids = audio_values = None

        if text is not None:
            tokens         = self.text_encoder.tokenize([text], device=device)
            input_ids      = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            token_type_ids = tokens.get("token_type_ids")

        if audio is not None:
            audio_values = self.audio_encoder.preprocess(audio, device=device)

        return self.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, audio_values=audio_values,
        )

    @torch.no_grad()
    def predict_stream_chunk(
        self,
        chunk:  np.ndarray,
        text:   Optional[str] = None,
        device: str = "cpu",
    ) -> Dict:
        self.eval()
        audio_values = self.audio_encoder.preprocess_stream_chunk(chunk, device=device)
        input_ids = attention_mask = token_type_ids = None

        if text is not None:
            tokens         = self.text_encoder.tokenize([text], device=device)
            input_ids      = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            token_type_ids = tokens.get("token_type_ids")

        return self.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, audio_values=audio_values,
        )