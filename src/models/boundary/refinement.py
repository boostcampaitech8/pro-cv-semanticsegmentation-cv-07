# src/models/boundary/refinement.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import EdgeAwareRefine


class BoundaryRefinementModule(nn.Module):
    """
    Boundary 기반 segmentation refinement
    - 전체 재예측 ❌
    - 경계 주변만 선택적 재판단 ⭕
    """

    def __init__(self, num_classes):
        super().__init__()
        self.refine = EdgeAwareRefine(num_classes)

    def forward(self, seg_logits, boundary_logits):
        """
        seg_logits: (B, C, H, W)
        boundary_logits: (B, 1, H, W)
        """
        boundary_prob = torch.sigmoid(boundary_logits)

        # 경계 강조 (soft mask)
        edge_mask = boundary_prob

        refined_seg = self.refine(seg_logits, edge_mask)
        return refined_seg