"""
Text Encoder for CLIP contrastive learning.

Takes pre-computed text embeddings (e.g., 128-dim vectors from OpenAI API stored
in caption.json) and projects them into the shared CLIP embedding space via a
trainable MLP projection head.

This is NOT a raw-text encoder — the input is already a dense embedding vector.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Encodes pre-computed text embeddings into the shared CLIP embedding space.

    Architecture:
        Linear → GELU → Linear → LayerNorm

    Args:
        input_dim: Dimension of the input text embedding (default: 128).
        clip_dim: Dimension of the shared CLIP embedding space.
        hidden_dim: Hidden dimension of the projection MLP.
    """

    def __init__(self, input_dim: int = 128, clip_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embedding: Pre-computed text embedding of shape [B, input_dim].

        Returns:
            Tensor of shape [B, clip_dim].
        """
        return self.projection(text_embedding)
