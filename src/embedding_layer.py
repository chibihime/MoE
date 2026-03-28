"""
embedding_layer.py
==================
Multimodal Embedding Layer for DEAP Emotion Recognition (MoE Architecture)

Architecture Overview:
----------------------
- PhysioEncoder  : 2-layer MLP for physiological features (32-dim: 8ch × {mean,std,min,max})
- AudioEncoder   : 2-layer MLP for audio/MIDI features (3-dim: note_density, mean_pitch, tempo)
- ContextEncoder : Learns a modality-availability context vector from a binary presence mask

Missing Modality Handling:
--------------------------
Each encoder owns a *learned mask token* — a trainable nn.Parameter of the same shape as
the encoder's output embedding. When a modality is absent for a given sample, the mask token
replaces that sample's embedding. This lets the network learn a meaningful "this modality is
missing" representation rather than collapsing to zeros.

Output:
-------
The layer returns a dict of separate per-modality embeddings (not fused). This is intentional:
the downstream MoE Soft-Gating Network needs independent embeddings to compute modality weights.

    {
        "physio"  : Tensor [B, embed_dim],
        "audio"   : Tensor [B, embed_dim],
        "context" : Tensor [B, embed_dim],
    }

Usage Example:
--------------
    import torch
    from embedding_layer import MultimodalEmbeddingLayer

    model = MultimodalEmbeddingLayer(
        physio_input_dim=32,   # 8 channels × 4 stats
        audio_input_dim=3,     # note_density, mean_pitch, tempo
        embed_dim=64,
        dropout=0.1,
    )

    B = 16  # batch size
    physio_feats  = torch.randn(B, 32)
    audio_feats   = torch.randn(B, 3)
    # Binary mask: 1 = modality present, 0 = missing  shape [B, 2]
    # Column 0 → physio, Column 1 → audio
    modality_mask = torch.ones(B, 2)
    modality_mask[0, 1] = 0   # first sample has no audio

    embeddings = model(physio_feats, audio_feats, modality_mask)
    # embeddings["physio"].shape  → [16, 64]
    # embeddings["audio"].shape   → [16, 64]
    # embeddings["context"].shape → [16, 64]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Helper: MLP block with BatchNorm + optional Dropout
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    """
    A single fully-connected block:
        Linear → BatchNorm1d → ReLU → (Dropout)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Physiological Encoder
# ---------------------------------------------------------------------------

class PhysioEncoder(nn.Module):
    """
    MLP encoder for physiological features.

    Input  : [B, physio_input_dim]   (default 32: 8ch × {mean, std, min, max})
    Output : [B, embed_dim]

    Architecture:
        physio_input_dim → hidden_dim → embed_dim
    Two MLPBlocks; second block has no BatchNorm because it is the output projection.

    Missing modality: replaced by a learned mask token (nn.Parameter).
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(input_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Learned token representing "physiological modality absent"
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                        # [B, input_dim]
        present_mask: Optional[torch.Tensor],   # [B] bool/float, 1=present 0=absent
    ) -> torch.Tensor:                          # [B, embed_dim]
        emb = self.net(x)                       # [B, embed_dim]

        if present_mask is not None:
            # Broadcast mask_token to batch; replace absent samples
            mask = present_mask.unsqueeze(-1).float()   # [B, 1]
            token = self.mask_token.expand(x.size(0), -1)  # [B, embed_dim]
            emb = mask * emb + (1.0 - mask) * token

        return emb


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    MLP encoder for audio/MIDI features.

    Input  : [B, audio_input_dim]   (default 3: note_density, mean_pitch, tempo)
    Output : [B, embed_dim]

    Architecture:
        audio_input_dim → hidden_dim → embed_dim

    Missing modality: replaced by a learned mask token (nn.Parameter).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(input_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Learned token representing "audio modality absent"
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                        # [B, input_dim]
        present_mask: Optional[torch.Tensor],   # [B] bool/float
    ) -> torch.Tensor:                          # [B, embed_dim]
        emb = self.net(x)

        if present_mask is not None:
            mask = present_mask.unsqueeze(-1).float()
            token = self.mask_token.expand(x.size(0), -1)
            emb = mask * emb + (1.0 - mask) * token

        return emb


# ---------------------------------------------------------------------------
# Context Encoder
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    Encodes *which* modalities are present into a context embedding.

    Input  : [B, num_modalities]  — binary presence mask (float)
    Output : [B, embed_dim]

    Rationale: The MoE gating network benefits from an explicit signal about
    which modalities are available.  Rather than hard-coding this as a rule,
    we let a small MLP learn the mapping from presence patterns to a context
    vector that the gating network can condition on.
    """

    def __init__(
        self,
        num_modalities: int = 2,    # physio + audio
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(num_modalities, 16, dropout=0.0),   # small; no dropout here
            nn.Linear(16, embed_dim),
        )

    def forward(self, modality_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_mask : [B, num_modalities] — 1.0 present, 0.0 absent
        Returns:
            context_emb   : [B, embed_dim]
        """
        return self.net(modality_mask.float())


# ---------------------------------------------------------------------------
# Multimodal Embedding Layer  (top-level module)
# ---------------------------------------------------------------------------

class MultimodalEmbeddingLayer(nn.Module):
    """
    Combines PhysioEncoder, AudioEncoder, and ContextEncoder into a single
    module that produces three aligned embeddings ready for MoE fusion.

    Args:
        physio_input_dim  : Dimensionality of physiological features (default 32).
        audio_input_dim   : Dimensionality of audio features (default 3).
        embed_dim         : Shared embedding size for all modalities (default 64).
        physio_hidden_dim : Hidden units inside PhysioEncoder (default 128).
        audio_hidden_dim  : Hidden units inside AudioEncoder (default 32).
        dropout           : Dropout probability in encoder MLPs (default 0.1).
        num_modalities    : Number of modalities tracked in the presence mask (default 2).

    Forward Args:
        physio_feats  : [B, physio_input_dim] — physiological feature vectors.
        audio_feats   : [B, audio_input_dim]  — audio feature vectors.
        modality_mask : [B, num_modalities]   — 1.0 if modality present, 0.0 if absent.
                        Column 0 → physio, Column 1 → audio.

    Returns:
        dict with keys "physio", "audio", "context", each [B, embed_dim].
    """

    def __init__(
        self,
        physio_input_dim: int = 32,
        audio_input_dim: int = 3,
        embed_dim: int = 64,
        physio_hidden_dim: int = 128,
        audio_hidden_dim: int = 32,
        dropout: float = 0.1,
        num_modalities: int = 2,
    ):
        super().__init__()

        self.physio_encoder = PhysioEncoder(
            input_dim=physio_input_dim,
            hidden_dim=physio_hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.audio_encoder = AudioEncoder(
            input_dim=audio_input_dim,
            hidden_dim=audio_hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.context_encoder = ContextEncoder(
            num_modalities=num_modalities,
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        physio_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            physio_feats  : [B, physio_input_dim]
            audio_feats   : [B, audio_input_dim]
            modality_mask : [B, 2]  (col 0 = physio, col 1 = audio)

        Returns:
            {
                "physio"  : [B, embed_dim],
                "audio"   : [B, embed_dim],
                "context" : [B, embed_dim],
            }
        """
        physio_present = modality_mask[:, 0]   # [B]
        audio_present  = modality_mask[:, 1]   # [B]

        physio_emb  = self.physio_encoder(physio_feats, physio_present)
        audio_emb   = self.audio_encoder(audio_feats, audio_present)
        context_emb = self.context_encoder(modality_mask)

        return {
            "physio":  physio_emb,
            "audio":   audio_emb,
            "context": context_emb,
        }


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly: python embedding_layer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B = 8   # batch size

    model = MultimodalEmbeddingLayer(
        physio_input_dim=32,
        audio_input_dim=3,
        embed_dim=64,
        dropout=0.1,
    )
    model.eval()

    physio_feats  = torch.randn(B, 32)
    audio_feats   = torch.randn(B, 3)

    # Simulate: first two samples are missing audio
    modality_mask = torch.ones(B, 2)
    modality_mask[0, 1] = 0
    modality_mask[1, 1] = 0

    with torch.no_grad():
        embeddings = model(physio_feats, audio_feats, modality_mask)

    print("=== Embedding Layer Smoke Test ===")
    for name, emb in embeddings.items():
        print(f"  {name:8s} embedding shape: {tuple(emb.shape)}")

    # Verify that missing-audio samples get identical audio embeddings (mask token)
    audio_emb = embeddings["audio"]
    assert torch.allclose(audio_emb[0], audio_emb[1]), \
        "Samples with missing audio should share the learned mask token."
    print("\n  ✓ Missing modality mask token applied correctly.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total trainable parameters: {total_params:,}")