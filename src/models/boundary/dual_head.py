# src/models/boundary/dual_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import segmentation_models_pytorch as smp

from .decoder_factory import build_unet_decoder


class DualHeadBoundaryNet(nn.Module):
    """
    Dual-Head Boundary-Segmentation Network

    - Shared encoder
    - Two decoders:
        1) Segmentation head
        2) Boundary head
    - Boundary feature injected into segmentation logits
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 29,
    ):
        super().__init__()

        # -------------------------
        # Encoder (shared)
        # -------------------------
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        encoder_channels = self.encoder.out_channels  # [C1, C2, C3, C4, C5]

        # -------------------------
        # Segmentation decoder
        # -------------------------
        self.seg_decoder = build_unet_decoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_bn=False,
            center=False,
        )

        self.seg_head = nn.Conv2d(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=1,
        )

        # -------------------------
        # Boundary decoder (lighter)
        # -------------------------
        self.boundary_decoder = build_unet_decoder(
            encoder_channels=encoder_channels,
            decoder_channels=(128, 64, 32, 16, 8),
            n_blocks=5,
            use_bn=False,
            center=False,
        )

        self.boundary_head = nn.Conv2d(
            in_channels=8,
            out_channels=1,  # binary boundary map
            kernel_size=1,
        )

        # -------------------------
        # Boundary → Segmentation injection
        # -------------------------
        self.refine_conv = nn.Conv2d(num_classes + 1, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # -------------------------
        # Encoder
        # -------------------------
        features = self.encoder(x)

        # -------------------------
        # Decoders
        # -------------------------
        seg_feat = self.seg_decoder(features)
        bnd_feat = self.boundary_decoder(features)

        seg_logits = self.seg_head(seg_feat)
        boundary_logits = self.boundary_head(bnd_feat)

        # -------------------------
        # Boundary → Segmentation refinement
        # -------------------------
        boundary_prob = torch.sigmoid(boundary_logits)

        if boundary_prob.shape[-2:] != seg_logits.shape[-2:]:
            boundary_prob = F.interpolate(
                boundary_prob,
                size=seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat([seg_logits, 0.1 * boundary_prob], dim=1)
        refined_seg = seg_logits + self.refine_conv(fused)   # residual

        return {
            "seg": refined_seg,
            "boundary": boundary_logits,
        }