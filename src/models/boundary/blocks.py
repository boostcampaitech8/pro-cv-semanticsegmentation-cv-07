# src/models/boundary/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Boundary 기반 refinement에서 사용하는 기본 블록들.
- boundary confidence로 feature를 조절하거나
- 경계 주변만 선택적으로 refinement 수행
"""
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class BoundaryConfidenceGate(nn.Module):
    """
    Boundary probability를 이용해 feature map을 soft-gating.
    - boundary가 강한 위치에서만 feature를 강조
    """
    def __init__(self):
        super().__init__()

    def forward(self, feat, boundary_prob):
        if boundary_prob.shape[-2:] != feat.shape[-2:]:
            boundary_prob = F.interpolate(
                boundary_prob,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return feat * boundary_prob


class EdgeAwareRefine(nn.Module):
    """
    Boundary 근처에서만 segmentation feature를 보정.
    - 전체 재예측 x
    - 경계 영역만 선택적 refinement
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x, edge_mask):
        if edge_mask.shape[-2:] != x.shape[-2:]:
            edge_mask = F.interpolate(edge_mask, x.shape[-2:], mode="bilinear")

        refined = self.conv(x)
        return x + refined * edge_mask