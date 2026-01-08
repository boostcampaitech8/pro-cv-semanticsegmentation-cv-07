# src/models/boundary/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class BoundaryConfidenceGate(nn.Module):
    """
    Boundary confidence로 feature를 gating
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
    Boundary 주변에서만 refinement 수행
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