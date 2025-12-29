# src/models/build_model.py

import torch.nn as nn
from src.models.smp_model import build_smp_model
from src.models.boundary.dual_head import DualHeadBoundaryNet
from src.models.boundary.basnet_like import BASNetLike


class SegOnlyWrapper(nn.Module):
    """
    Boundary 모델을 기존 trainer와 호환시키기 위한 wrapper
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["seg"]


def build_model(cfg):
    # Baseline
    if cfg.boundary_mode == "none":
        return build_smp_model(cfg)

    encoder_weights = "imagenet" if cfg.pretrained == "imagenet" else None

    if cfg.boundary_mode == "dual":
        return DualHeadBoundaryNet(
            encoder_name=cfg.encoder_name,
            encoder_weights=encoder_weights,
        )

    elif cfg.boundary_mode == "basnet":
        return BASNetLike(
            encoder_name=cfg.encoder_name,
            encoder_weights=encoder_weights,
        )

    else:
        raise ValueError(f"Unknown boundary_mode: {cfg.boundary_mode}")

    # # 🔹 RADImageNet weight는 encoder에 수동 로딩
    # if cfg.pretrained == "radimagenet":
    #     rad_ckpt = "/path/to/RadImageNet-ResNet50.pth"
    #     state = torch.load(rad_ckpt, map_location="cpu")
    #     model.encoder.load_state_dict(state, strict=False)