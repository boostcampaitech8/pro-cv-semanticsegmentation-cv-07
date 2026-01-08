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
    Boundary-aware stream + Semantic-aware stream

    설계 의도 :
    - encoder / decoder를 공유하여 연산효율 유지
    - boundary branch는 보조 신호(auxiliary signal) 역할
    - boundary는 독립적인 출력이 아니라, segmentation을 refinement하기 위한 공간적 힌트로 사용
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
        if isinstance(self.base_model.decoder, (smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder,)):
            dec = self.decoder(feats)
        else:
            dec = self.decoder(*feats)

        seg_logits = self.seg_head(dec)
        boundary_feat = dec.detach() if self.detach_boundary else dec
        # NOTE:
        # boundary logits는 segmentation을 직접 예측하지 않음
        # 경계 위치의 불확실성을 표현하는 신호로만 사용됨
        boundary_logits = self.boundary_head(boundary_feat)

        if self.use_refinement:
            # Boundary 기반 refinement
            # 전체 재예측이 아니라, 경계 근처에서만 segmentation 보정
            seg_logits = self.boundary_refine(seg_logits, boundary_logits)

        return {
            "seg": seg_logits,
            "boundary": boundary_logits,
        }
