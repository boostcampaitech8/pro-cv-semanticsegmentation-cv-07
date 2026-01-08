# src/models/boundary/basnet_like.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .blocks import BoundaryConfidenceGate
from .refinement import BoundaryRefinementModule
from .decoder_factory import build_unet_decoder
from .transformer_mediator import BoundaryMediatorTransformer
# from .logit_scaler import ClassWiseLogitScaler



class BASNetLike(nn.Module):
    """
    SMP segmentation model 위에 boundary head를 추가한 구조.
    - semantic branch: 기존 segmentation
    - boundary branch: decoder feature 기반 경계 예측
    - refinement: boundary를 이용해 seg logit만 선택적으로 보정
    """

    def __init__(self, base_model, num_classes: int, use_refinement=True, use_transformer=False, detach_boundary=True):
        super().__init__()
        self.base_model = base_model
        self.decoder_boundary = detach_boundary

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.seg_head = base_model.segmentation_head

        decoder_out_ch = self.decoder.out_channels[-1]

        self.boundary_head = nn.Conv2d(
            decoder_out_ch, 1, kernel_size=1
        )

        self.use_refinement = use_refinement
        if use_refinement:
            self.boundary_refine = BoundaryRefinementModule(
                num_classes=self.seg_head.out_channels
            )

    def forward(self, x):
        feats = self.encoder(x)
        # Unet++ decoder는 feature list를 그대로 받음
        # (일반 SMP decoder는 *feats 형태)
        if isinstance(self.base_model.decoder, (smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder,)):
            dec = self.decoder(feats)
        else:
            dec = self.decoder(*feats)

        seg_logits = self.seg_head(dec)
        boundary_feat = dec.detach() if self.detach_boundary else dec
        # boundary logit은 decoder feature에서만 생성
        boundary_logits = self.boundary_head(boundary_feat)

        # refinement는 segmentation logit만 수정
        # boundary는 loss / gate 용도로만 사용
        if self.use_refinement:
            seg_logits = self.boundary_refine(seg_logits, boundary_logits)

        return {
            "seg": seg_logits,
            "boundary": boundary_logits,
        }
