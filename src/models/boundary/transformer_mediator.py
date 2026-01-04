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
        B, C, H, W = feat.shape

        # 🔻 spatial downsample (ex: 8x)
        feat_ds = F.avg_pool2d(feat, kernel_size=16, stride=16)
        boundary_ds = F.avg_pool2d(boundary_logits, kernel_size=16, stride=16)

        Bd, Cd, Hd, Wd = feat_ds.shape  # Hd*Wd ≈ 4096

        boundary_prob = torch.sigmoid(boundary_ds).view(B, 1, -1)

        x = feat_ds.view(B, C, -1).permute(0, 2, 1)  # (B, 4096, C)

        attn_out, _ = self.attn(x, x, x)

        attn_out = attn_out * boundary_prob.transpose(1, 2)

        out = self.norm(x + attn_out)
        out = out.permute(0, 2, 1).view(B, C, Hd, Wd)

        # 🔺 upsample back
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out