"""
moe_fusion_layer.py
====================
Mixture-of-Experts (MoE) Fusion Layer for DEAP Continuous Emotion Prediction

Architecture Overview:
-----------------------
This module sits directly downstream of MultimodalEmbeddingLayer and is
responsible for fusing the three per-modality embeddings into continuous
predictions for all four DEAP emotion dimensions:

    Valence · Arousal · Dominance · Liking

Design Decisions:
-----------------
1. **Soft Gating** (not sparse Top-K):
   DEAP is a small dataset (~1280 samples). Sparse gating risks dead experts
   and requires auxiliary load-balancing losses to stabilize training.
   Soft gating is fully differentiable, numerically stable, and trains well
   on small data — every expert always contributes, weighted by learned gates.

2. **Four Experts**:
   - PhysioExpert    : Processes the physio embedding alone.
   - AudioExpert     : Processes the audio embedding alone.
   - CrossModalExpert: Processes the *concatenation* of physio + audio embeddings,
                       learning cross-modal interactions.
   - ContextExpert   : Processes the context embedding, which already encodes
                       modality availability — anchors the MoE in the presence pattern.

3. **Context-Conditioned Gating**:
   The gating network takes the context embedding as input. Since the context
   embedding is derived from the binary modality-presence mask, the gate
   automatically learns to re-weight experts depending on which modalities
   are available (e.g., suppress AudioExpert when audio is missing).

4. **Multi-Task Output Heads**:
   Four independent linear heads share the fused representation. Each head
   predicts one continuous score in [0, 9] (DEAP rating scale).
   Using separate heads allows each dimension to develop its own projection
   while sharing the fused features.

5. **Load Balance Monitoring**:
   `load_balance_loss()` computes the auxiliary load-balance loss from
   Shazeer et al. (2017). For soft gating this is optional to include in
   training, but it is useful to monitor expert utilization and can be
   added as a small regularisation term (λ ~ 0.01).

Input (from MultimodalEmbeddingLayer):
---------------------------------------
    embeddings = {
        "physio"  : Tensor [B, embed_dim],
        "audio"   : Tensor [B, embed_dim],
        "context" : Tensor [B, embed_dim],
    }

Output:
-------
    {
        "valence"   : Tensor [B, 1],
        "arousal"   : Tensor [B, 1],
        "dominance" : Tensor [B, 1],
        "liking"    : Tensor [B, 1],
        "gate_weights": Tensor [B, num_experts],   # for logging / analysis
    }

Usage Example:
--------------
    import torch
    from embedding_layer import MultimodalEmbeddingLayer
    from moe_fusion_layer import MoEFusionLayer

    embed_dim = 64
    B = 16

    emb_layer = MultimodalEmbeddingLayer(embed_dim=embed_dim)
    moe_layer = MoEFusionLayer(embed_dim=embed_dim)

    physio = torch.randn(B, 32)
    audio  = torch.randn(B, 3)
    mask   = torch.ones(B, 2)
    mask[0, 1] = 0  # first sample: no audio

    embeddings = emb_layer(physio, audio, mask)
    predictions = moe_layer(embeddings)

    # predictions["valence"].shape  → [16, 1]
    # predictions["gate_weights"].shape → [16, 4]

    # Optional: add to training loss for expert utilization regularisation
    lb_loss = moe_layer.load_balance_loss(predictions["gate_weights"])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from embedding_layer import MLPBlock


# ---------------------------------------------------------------------------
# Individual Experts
# ---------------------------------------------------------------------------

class PhysioExpert(nn.Module):
    """
    Processes the physiological embedding.

    Architecture: embed_dim → expert_hidden_dim → embed_dim
    Uses two MLPBlocks with residual connection for stable gradient flow.
    """

    def __init__(self, embed_dim: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(embed_dim, expert_hidden_dim, dropout=dropout),
            nn.Linear(expert_hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, physio_emb: torch.Tensor) -> torch.Tensor:
        # Residual: helps retain modality-specific signal after projection
        return self.norm(physio_emb + self.net(physio_emb))


class AudioExpert(nn.Module):
    """
    Processes the audio embedding.

    Architecture mirrors PhysioExpert for symmetry, with its own
    independent weights.
    """

    def __init__(self, embed_dim: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(embed_dim, expert_hidden_dim, dropout=dropout),
            nn.Linear(expert_hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, audio_emb: torch.Tensor) -> torch.Tensor:
        return self.norm(audio_emb + self.net(audio_emb))


class CrossModalExpert(nn.Module):
    """
    Processes the *concatenation* of physio + audio embeddings to learn
    cross-modal interactions (e.g., physiological arousal correlating with
    high note density / fast tempo).

    Input  : [B, 2 * embed_dim]
    Output : [B, embed_dim]

    No residual here since the input dimension differs from the output.
    """

    def __init__(self, embed_dim: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(2 * embed_dim, expert_hidden_dim, dropout=dropout),
            nn.Linear(expert_hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, physio_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([physio_emb, audio_emb], dim=-1)  # [B, 2*embed_dim]
        return self.norm(self.net(x))


class ContextExpert(nn.Module):
    """
    Processes the context embedding, which encodes *which* modalities are
    present. This expert acts as an adaptive prior — when both modalities
    are available it learns enriched shared patterns; when one is missing
    it learns a compensatory representation.

    Architecture mirrors PhysioExpert.
    """

    def __init__(self, embed_dim: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(embed_dim, expert_hidden_dim, dropout=dropout),
            nn.Linear(expert_hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        return self.norm(context_emb + self.net(context_emb))


# ---------------------------------------------------------------------------
# Context-Conditioned Soft Gating Network
# ---------------------------------------------------------------------------

class SoftGatingNetwork(nn.Module):
    """
    Produces soft mixture weights over the four experts.

    Input  : context_emb [B, embed_dim]
              — derived from the binary modality-presence mask, so the
                gate learns to suppress experts for absent modalities
                without any explicit rule.
    Output : gate_weights [B, num_experts]  — sums to 1 per sample.

    Architecture:
        embed_dim → gate_hidden_dim → num_experts  (+ Softmax)

    Note: We use a dedicated hidden layer rather than a direct linear
    projection so the gate can learn non-linear threshold behaviour
    (e.g., sharply down-weighting AudioExpert when audio is absent).
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        gate_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            MLPBlock(embed_dim, gate_hidden_dim, dropout=dropout),
            nn.Linear(gate_hidden_dim, num_experts),
        )

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_emb : [B, embed_dim]
        Returns:
            gate_weights : [B, num_experts]  — softmax-normalised
        """
        logits = self.net(context_emb)              # [B, num_experts]
        return F.softmax(logits, dim=-1)            # [B, num_experts]


# ---------------------------------------------------------------------------
# Multi-Task Output Heads
# ---------------------------------------------------------------------------

class EmotionOutputHeads(nn.Module):
    """
    Four independent linear projection heads, one per DEAP emotion dimension.

    Each head maps the fused [B, embed_dim] representation to a single
    continuous score. Using separate heads lets each dimension develop its
    own projection while sharing the upstream fused features.

    DEAP ratings are in [1, 9]; we do *not* clamp here — let the loss
    function (e.g. MSE) drive the range during training.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.valence   = nn.Linear(embed_dim, 1)
        self.arousal   = nn.Linear(embed_dim, 1)
        self.dominance = nn.Linear(embed_dim, 1)
        self.liking    = nn.Linear(embed_dim, 1)

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused : [B, embed_dim]
        Returns:
            dict with keys valence, arousal, dominance, liking → each [B, 1]
        """
        return {
            "valence"  : self.valence(fused),
            "arousal"  : self.arousal(fused),
            "dominance": self.dominance(fused),
            "liking"   : self.liking(fused),
        }


# ---------------------------------------------------------------------------
# MoE Fusion Layer  (top-level module)
# ---------------------------------------------------------------------------

class MoEFusionLayer(nn.Module):
    """
    Mixture-of-Experts fusion layer that takes the three per-modality
    embeddings from MultimodalEmbeddingLayer and produces continuous
    predictions for all four DEAP emotion dimensions.

    Args:
        embed_dim        : Must match the embed_dim used in MultimodalEmbeddingLayer.
        expert_hidden_dim: Hidden units inside each expert MLP (default 128).
        gate_hidden_dim  : Hidden units inside the gating network (default 64).
        dropout          : Dropout probability in expert + gate MLPs (default 0.1).

    Forward Args:
        embeddings : dict returned by MultimodalEmbeddingLayer.forward()
                     Keys: "physio", "audio", "context" — each [B, embed_dim].

    Returns:
        dict with keys:
            "valence"      : [B, 1]
            "arousal"      : [B, 1]
            "dominance"    : [B, 1]
            "liking"       : [B, 1]
            "gate_weights" : [B, 4]  — softmax weights (physio, audio, cross, context)
    """

    NUM_EXPERTS = 4
    EXPERT_NAMES = ["physio", "audio", "cross_modal", "context"]

    def __init__(
        self,
        embed_dim: int = 64,
        expert_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- Experts ---
        self.physio_expert = PhysioExpert(
            embed_dim=embed_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )
        self.audio_expert = AudioExpert(
            embed_dim=embed_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )
        self.cross_modal_expert = CrossModalExpert(
            embed_dim=embed_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )
        self.context_expert = ContextExpert(
            embed_dim=embed_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )

        # --- Gating network (context-conditioned) ---
        self.gate = SoftGatingNetwork(
            embed_dim=embed_dim,
            num_experts=self.NUM_EXPERTS,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )

        # --- Output heads ---
        self.output_heads = EmotionOutputHeads(embed_dim=embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings : {
                "physio"  : [B, embed_dim],
                "audio"   : [B, embed_dim],
                "context" : [B, embed_dim],
            }

        Returns:
            {
                "valence"      : [B, 1],
                "arousal"      : [B, 1],
                "dominance"    : [B, 1],
                "liking"       : [B, 1],
                "gate_weights" : [B, 4],
            }
        """
        physio_emb  = embeddings["physio"]    # [B, embed_dim]
        audio_emb   = embeddings["audio"]     # [B, embed_dim]
        context_emb = embeddings["context"]   # [B, embed_dim]

        # --- Expert outputs: each [B, embed_dim] ---
        e_physio  = self.physio_expert(physio_emb)
        e_audio   = self.audio_expert(audio_emb)
        e_cross   = self.cross_modal_expert(physio_emb, audio_emb)
        e_context = self.context_expert(context_emb)

        # Stack expert outputs: [B, num_experts, embed_dim]
        expert_stack = torch.stack(
            [e_physio, e_audio, e_cross, e_context], dim=1
        )

        # --- Gate: [B, num_experts] ---
        gate_weights = self.gate(context_emb)   # [B, 4]

        # --- Soft mixture: weighted sum over experts ---
        # gate_weights: [B, 4] → [B, 4, 1] for broadcasting
        w = gate_weights.unsqueeze(-1)              # [B, 4, 1]
        fused = (w * expert_stack).sum(dim=1)       # [B, embed_dim]

        # --- Emotion predictions ---
        return {
        "fused": fused,
        "gate_weights": gate_weights
    }
    # ------------------------------------------------------------------
    # Auxiliary: Load Balance Loss
    # ------------------------------------------------------------------

    @staticmethod
    def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Computes the auxiliary load-balance loss from Shazeer et al. (2017).

        Encourages roughly equal expert utilisation across a batch.
        For soft gating, add this to the training loss with a small coefficient
        λ ∈ [0.001, 0.05]:

            total_loss = task_loss + λ * moe.load_balance_loss(gate_weights)

        Args:
            gate_weights : [B, num_experts] — the "gate_weights" from forward().

        Returns:
            scalar Tensor — load balance loss.

        Formula:
            L_lb = num_experts · Σ_i (f_i · p_i)
            where:
                f_i = fraction of tokens routed to expert i (mean of hard argmax)
                p_i = mean soft gate weight for expert i
        """
        num_experts = gate_weights.size(-1)

        # Fraction of samples where each expert has the highest gate weight
        expert_indices = gate_weights.argmax(dim=-1)                    # [B]
        f = torch.zeros(num_experts, device=gate_weights.device)
        for i in range(num_experts):
            f[i] = (expert_indices == i).float().mean()                 # scalar

        # Mean soft gate weight per expert
        p = gate_weights.mean(dim=0)                                    # [num_experts]

        return num_experts * (f * p).sum()

    # ------------------------------------------------------------------
    # Utility: gate weight summary (useful during training / analysis)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gate_summary(self, gate_weights: torch.Tensor) -> Dict[str, float]:
        """
        Returns mean gate weight per expert over the batch.
        Useful for logging to tensorboard / wandb.

        Args:
            gate_weights : [B, num_experts]

        Returns:
            dict  { "gate/physio": float, "gate/audio": float,
                    "gate/cross_modal": float, "gate/context": float }
        """
        means = gate_weights.mean(dim=0).tolist()
        return {
            f"gate/{name}": val
            for name, val in zip(self.EXPERT_NAMES, means)
        }


# ---------------------------------------------------------------------------
# Quick smoke-test (python moe_fusion_layer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from embedding_layer import MultimodalEmbeddingLayer

    torch.manual_seed(42)

    EMBED_DIM = 64
    B = 8

    # --- Build both layers ---
    emb_layer = MultimodalEmbeddingLayer(
        physio_input_dim=32,
        audio_input_dim=3,
        embed_dim=EMBED_DIM,
        dropout=0.1,
    )
    moe_layer = MoEFusionLayer(
        embed_dim=EMBED_DIM,
        expert_hidden_dim=128,
        gate_hidden_dim=64,
        dropout=0.1,
    )
    emb_layer.eval()
    moe_layer.eval()

    # --- Synthetic inputs ---
    physio = torch.randn(B, 32)
    audio  = torch.randn(B, 3)

    # Samples 0 & 1: audio missing; sample 2: physio missing
    modality_mask = torch.ones(B, 2)
    modality_mask[0, 1] = 0
    modality_mask[1, 1] = 0
    modality_mask[2, 0] = 0

    with torch.no_grad():
        embeddings  = emb_layer(physio, audio, modality_mask)
        predictions = moe_layer(embeddings)

    print("=== MoE Fusion Layer Smoke Test ===\n")

    # --- Output shapes ---
    print("Output shapes:")
    for key, val in predictions.items():
        print(f"  {key:15s}: {tuple(val.shape)}")

    # --- Gate weight summary ---
    print("\nMean gate weights (full batch):")
    summary = moe_layer.gate_summary(predictions["gate_weights"])
    for name, val in summary.items():
        bar = "█" * int(val * 40)
        print(f"  {name:22s}: {val:.4f}  {bar}")

    # --- Gate differences for missing-audio samples vs full samples ---
    gate_w = predictions["gate_weights"]
    no_audio_gate = gate_w[:2].mean(0)    # samples 0 & 1: no audio
    full_gate     = gate_w[3:].mean(0)    # samples 3-7: both modalities

    print("\nGate weights — missing audio samples (mean):")
    for name, val in zip(MoEFusionLayer.EXPERT_NAMES, no_audio_gate.tolist()):
        print(f"  {name:15s}: {val:.4f}")

    print("\nGate weights — full modality samples (mean):")
    for name, val in zip(MoEFusionLayer.EXPERT_NAMES, full_gate.tolist()):
        print(f"  {name:15s}: {val:.4f}")

    # --- Load balance loss ---
    lb = MoEFusionLayer.load_balance_loss(predictions["gate_weights"])
    print(f"\nLoad balance loss (untrained, expect ~1.0): {lb.item():.4f}")

    # --- Parameter counts ---
    emb_params = sum(p.numel() for p in emb_layer.parameters() if p.requires_grad)
    moe_params = sum(p.numel() for p in moe_layer.parameters() if p.requires_grad)
    print(f"\nTrainable parameters:")
    print(f"  Embedding layer : {emb_params:,}")
    print(f"  MoE fusion layer: {moe_params:,}")
    print(f"  Total           : {emb_params + moe_params:,}")