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
    # 1️⃣ Baseline
    if cfg.boundary_mode == "none":
        return build_smp_model(cfg)

    # 2️⃣ Dual-head
    if cfg.boundary_mode == "dual":
        model = DualHeadBoundaryNet(
            encoder_name=cfg.encoder_name,
            encoder_weights="imagenet" if cfg.pretrained else None,
        )

    # 3️⃣ BASNet
    elif cfg.boundary_mode == "basnet":
        model = BASNetLike(
            encoder_name=cfg.encoder_name,
            encoder_weights="imagenet" if cfg.pretrained else None,
        )
    else:
        raise ValueError(f"Unknown boundary_mode: {cfg.boundary_mode}")

    # 🔹 현재 trainer / loss / inference는 tensor를 기대함
    return SegOnlyWrapper(model)