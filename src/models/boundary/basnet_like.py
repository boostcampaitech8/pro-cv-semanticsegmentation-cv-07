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

    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=29,
        use_refinement=True,
        use_transformer=False
    ):
        super().__init__()
        self.use_refinement = use_refinement
        self.use_transformer = use_transformer

        # Shared encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        enc_channels = self.encoder.out_channels

        # Semantic decoder
        self.semantic_decoder = build_unet_decoder(
            encoder_channels=enc_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_bn=False,
            center=False,
        )
        self.semantic_head = nn.Conv2d(16, num_classes, 1)

        # Boundary decoder
        self.boundary_decoder = build_unet_decoder(
            encoder_channels=enc_channels,
            decoder_channels=(128, 64, 32, 16, 8),
            n_blocks=5,
            use_bn=False,
            center=False,
        )
        self.boundary_head = nn.Conv2d(8, 1, 1)

        # Cross refinement
        self.boundary_refine = BoundaryRefinementModule(num_classes)
        # self.conf_gate = BoundaryConfidenceGate()
        # NOTE: boundary gating is not used in current experiments

        if self.use_transformer:
            self.mediator = BoundaryMediatorTransformer(dim=16)

        # self.logit_scaler = ClassWiseLogitScaler(num_classes)


    def forward(self, x):
        feats = self.encoder(x)

        sem_feat = self.semantic_decoder(feats)
        bnd_feat = self.boundary_decoder(feats)

        boundary_logits = self.boundary_head(bnd_feat)

        # 1️⃣ Transformer는 feature 공간에서
        if self.use_transformer:
            sem_feat = self.mediator(sem_feat, boundary_logits)

        seg_logits = self.semantic_head(sem_feat)

        # 2️⃣ Refinement는 logits에서
        if self.use_refinement:
            seg_logits = self.boundary_refine(seg_logits, boundary_logits)

        # ✅ class-wise logit scaling (핵심)
        # seg_logits = self.logit_scaler(seg_logits)

        return {
            "seg": seg_logits,
            "boundary": boundary_logits,
}
