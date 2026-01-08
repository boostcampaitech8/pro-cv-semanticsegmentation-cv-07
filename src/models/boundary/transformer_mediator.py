# src/models/boundary/transformer_mediator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Boundary-aware class interaction module.
- segmentation feature 간의 class-level interaction을 transformer로 모델링
- boundary confidence를 attention gate로 사용
- 현재는 옵션(use_transformer)이지만,
  boundary 정보를 class 관계 학습으로 확장하기 위한 구조적 준비 단계
"""
class BoundaryMediatorTransformer(nn.Module):
    """
    Boundary confidence를 이용해 spatial token 간 attention을 조절하는 transformer.
    - boundary가 강한 위치에서 class interaction을 강화
    - full-resolution 사용 시 메모리 과다 → downsample 후 attention
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
        # spatial resolution을 줄여 attention token 수를 제한 (메모리 안정성)
        feat_ds = F.avg_pool2d(feat, kernel_size=16, stride=16)
        boundary_ds = F.avg_pool2d(boundary_logits, kernel_size=16, stride=16)

        Bd, Cd, Hd, Wd = feat_ds.shape  # Hd*Wd ≈ 4096
        # boundary confidence를 attention gate로 사용
        # 완전 차단 방지를 위해 clamp
        boundary_prob = torch.sigmoid(boundary_ds).view(B, 1, -1)
        boundary_prob = boundary_prob.clamp(0.1, 0.9)

        x = feat_ds.view(B, C, -1).permute(0, 2, 1)  # (B, 4096, C)

        attn_out, _ = self.attn(x, x, x)

        alpha = 0.5
        gate = 1.0 + alpha * boundary_prob.transpose(1, 2)
        attn_out = attn_out * gate

        out = self.norm(x + attn_out)
        out = out.permute(0, 2, 1).view(B, C, Hd, Wd)

        # 🔺 upsample back
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out