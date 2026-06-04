"""
CLIP projection heads for aligning LA Encoder latents with text embeddings.

Provides:
    - LAEncoderCLIPHead: Projects the LA Encoder output [B, C, T] → [B, clip_dim]
      using global average pooling followed by a linear projection.
    - clip_loss: Symmetric InfoNCE loss (CLIP-style contrastive loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAEncoderCLIPHead(nn.Module):
    """
    Attention-based CLIP projection head.
    Uses positional embeddings, a learnable [CLS] token, and a Multi-Head Self-Attention
    Transformer Encoder layer to aggregate temporal sequence features [B, C, T] -> [B, clip_dim].
    """
    def __init__(self, embedding_dim: int = 64, clip_dim: int = 256, num_heads: int = 4, num_layers: int = 2, max_len: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Learnable Positional Embeddings (to support sequence length up to max_len)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, embedding_dim))
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # High-capacity MLP Projection Head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, clip_dim),
            nn.LayerNorm(clip_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: LA Encoder output of shape [B, embedding_dim, flow_dim].

        Returns:
            Tensor of shape [B, clip_dim].
        """
        # z shape: [B, C, T] (where C is embedding_dim, T is flow_dim / temporal length)
        B, C, T = z.shape
        
        # Permute to batch-first sequence layout: [B, T, C]
        x = z.permute(0, 2, 1)
        
        # Prepend the learnable [CLS] token to the sequence: [B, 1, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, T + 1, C]
        
        # Add corresponding positional embeddings (sliced or dynamically interpolated if out-of-bounds)
        if T + 1 <= self.pos_embed.size(1):
            pos_emb = self.pos_embed[:, :T + 1, :]
        else:
            # Dynamically interpolate positional embeddings to fit the sequence length
            pos_emb = F.interpolate(
                self.pos_embed.permute(0, 2, 1),  # [1, C, max_len + 1]
                size=T + 1,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)  # [1, T + 1, C]
            
        x = x + pos_emb
        
        # Pass through Transformer Self-Attention layers
        x = self.transformer(x)  # Shape: [B, T + 1, C]
        
        # Extract the aggregated representation from the [CLS] token
        z_pooled = x[:, 0, :]  # Shape: [B, C]
        
        # Project to the target CLIP embedding space
        return self.projection(z_pooled)  # Shape: [B, clip_dim]


def clip_loss(
    text_emb: torch.Tensor,
    la_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (CLIP-style contrastive loss) with duplicate caption handling.

    Computes cosine similarity between all text-latent pairs in the batch.
    If duplicate text embeddings (identical captions) are detected in the batch,
    it automatically applies Supervised Contrastive Loss to avoid the false-negative penalty.
    Otherwise, it falls back to standard symmetric InfoNCE cross-entropy loss.

    Args:
        text_emb: [B, clip_dim] text embeddings.
        la_emb:   [B, clip_dim] latent embeddings.
        temperature: Scaling factor for the logits.

    Returns:
        Scalar loss.
    """
    # L2 normalize
    text_emb = F.normalize(text_emb, dim=-1)
    la_emb = F.normalize(la_emb, dim=-1)

    # Cosine similarity matrix [B, B]
    logits = text_emb @ la_emb.T / temperature

    # Detect duplicate captions in the batch to avoid false negatives
    with torch.no_grad():
        sim_text = text_emb @ text_emb.T
        duplicate_mask = (sim_text > 0.95).float()
        
    # Check if there are actual duplicates (more than just the diagonal self-pairs)
    has_duplicates = duplicate_mask.sum() > duplicate_mask.size(0)

    if has_duplicates:
        # Supervised Contrastive Loss (handles duplicates perfectly)
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits_stable)
        denom = exp_logits.sum(dim=1, keepdim=True)
        log_prob = logits_stable - torch.log(denom + 1e-8)
        
        # Calculate loss in both directions (symmetric)
        num_pos_t2l = duplicate_mask.sum(dim=1, keepdim=True)
        loss_t2l = - (duplicate_mask * log_prob).sum(dim=1, keepdim=True) / num_pos_t2l
        
        # Latent to Text direction
        logits_l2t = logits.T
        logits_max_l2t, _ = torch.max(logits_l2t, dim=1, keepdim=True)
        logits_l2t_stable = logits_l2t - logits_max_l2t.detach()
        exp_logits_l2t = torch.exp(logits_l2t_stable)
        denom_l2t = exp_logits_l2t.sum(dim=1, keepdim=True)
        log_prob_l2t = logits_l2t_stable - torch.log(denom_l2t + 1e-8)
        
        duplicate_mask_l2t = duplicate_mask.T
        num_pos_l2t = duplicate_mask_l2t.sum(dim=1, keepdim=True)
        loss_l2t = - (duplicate_mask_l2t * log_prob_l2t).sum(dim=1, keepdim=True) / num_pos_l2t
        
        return (loss_t2l.mean() + loss_l2t.mean()) / 2
    else:
        # Standard InfoNCE CLIP loss (when all captions are unique)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2l = F.cross_entropy(logits, labels)
        loss_l2t = F.cross_entropy(logits.T, labels)
        return (loss_t2l + loss_l2t) / 2
