# src/models/boundary/refinement.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import EdgeAwareRefine

"""
Boundary-aware segmentation refinement module.
- boundary logit을 soft edge mask로 변환
- segmentation logit을 경계 주변에서만 보정
"""
class BoundaryRefinementModule(nn.Module):
    """
    Boundary 기반 segmentation refinement
    - 전체 재예측 x
    - 경계 주변만 선택적 재판단 o
    """

    def __init__(self, num_classes):
        super().__init__()
        self.refine = EdgeAwareRefine(num_classes)

    def forward(self, seg_logits, boundary_logits):
        """
        seg_logits: (B, C, H, W)
        boundary_logits: (B, 1, H, W)
        """
        # boundary logit → 확률 → edge mask
        # hard mask ❌, soft mask ⭕ (gradient 안정성)
        boundary_prob = torch.sigmoid(boundary_logits)

        # 경계 강조 (soft mask)
        edge_mask = boundary_prob
        
        # 경계 주변에서만 refinement 적용
        refined_seg = self.refine(seg_logits, edge_mask)
        return refined_seg