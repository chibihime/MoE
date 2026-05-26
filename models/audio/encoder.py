"""
models/audio/encoder.py

Audio encoder using facebook/wav2vec2-base.
Handles both real-time mic stream chunks and uploaded audio files.

NOTE: We use Wav2Vec2 for acoustic feature extraction (not transcription).
This captures paralinguistic cues — pitch, rhythm, energy — that carry
emotion signal independent of word content.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Optional, Dict, Union


class AudioEncoder(nn.Module):
    """
    Encodes raw audio waveforms into dense feature vectors.

    - Backbone: facebook/wav2vec2-base (CNN feature extractor + 12 transformer layers)
    - CNN feature extractor and first `freeze_layers` transformer layers are frozen.
    - Temporal mean pooling collapses variable-length audio to a fixed vector.
    - Supports both:
        * File mode:   full clip (up to 30s) preprocessed before forward pass
        * Stream mode: fixed-size chunks (e.g. 500ms) fed incrementally
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_name = config.get("model_name", "facebook/wav2vec2-base")
        self.hidden_size = config.get("hidden_size", 768)
        self.sample_rate = config.get("sample_rate", 16000)
        freeze_layers = config.get("freeze_layers", 6)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self._freeze_layers(freeze_layers)

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _freeze_layers(self, num_layers: int):
        # Always freeze the CNN feature extractor
        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad = False
        for layer in self.encoder.encoder.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        audio_values: torch.Tensor,                    # (batch, num_samples)
        attention_mask: Optional[torch.Tensor] = None, # (batch, num_samples)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_values:   (batch, num_samples) — raw waveform, 16kHz, float32
            attention_mask: (batch, num_samples) optional, for padded batches

        Returns dict with keys:
            features:        (batch, hidden_size)  — projected temporal pool, used by router/fusion
            sequence_output: (batch, time, hidden)  — frame-level outputs
            pooled:          (batch, hidden_size)  — raw pool before projection
        """
        outputs = self.encoder(
            input_values=audio_values,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (batch, time, hidden)
        pooled = hidden.mean(dim=1)         # (batch, hidden)
        return {
            "features": self.projection(pooled),
            "sequence_output": hidden,
            "pooled": pooled,
        }

    def preprocess(
        self,
        audio: Union[np.ndarray, list],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Preprocess raw audio array(s) to model input tensor.

        Args:
            audio:  np.ndarray of shape (num_samples,) for a single clip,
                    or list of np.ndarray for a batch.
            device: target device string.

        Returns:
            input_values tensor of shape (batch, num_samples).
        """
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values.to(device)

    def preprocess_stream_chunk(
        self,
        chunk: np.ndarray,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Preprocess a single real-time audio chunk (e.g. 500ms = 8000 samples at 16kHz).

        Args:
            chunk: np.ndarray of shape (num_samples,)

        Returns:
            input_values tensor of shape (1, num_samples).
        """
        return self.preprocess([chunk], device=device)