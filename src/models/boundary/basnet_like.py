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
        boundary_logits = self.boundary_head(boundary_feat)

        if self.use_refinement:
            seg_logits = self.boundary_refine(seg_logits, boundary_logits)

        return {
            "seg": seg_logits,
            "boundary": boundary_logits,
        }
