# src/models/boundary/transformer_mediator.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryMediatorTransformer(nn.Module):
    """
    Transformer that mediates class interaction
    using boundary confidence
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat, boundary_logits):
        """
        feat: (B, C, H, W)
        boundary_logits: (B, 1, H, W)
        """
        B, C, H, W = feat.shape

        boundary_prob = torch.sigmoid(boundary_logits)
        boundary_prob = boundary_prob.view(B, 1, -1)

        x = feat.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        attn_out, _ = self.attn(x, x, x)

        # Boundary confidence gating
        attn_out = attn_out * boundary_prob.transpose(1, 2)

        out = self.norm(x + attn_out)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out