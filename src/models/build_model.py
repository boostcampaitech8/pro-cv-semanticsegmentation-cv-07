from src.models.smp_model import build_smp_model
from src.models.boundary.dual_head import DualHeadBoundaryNet
from src.models.boundary.basnet_like import BASNetLike

def build_model(cfg):
    # baseline
    if cfg.boundary_mode == "none":
        return build_smp_model(cfg)

    encoder_weights = "imagenet" if cfg.pretrained == "imagenet" else None

    if cfg.boundary_mode == "dual":
        return DualHeadBoundaryNet(
            encoder_name=cfg.encoder_name,
            encoder_weights=encoder_weights,
        )

    if cfg.boundary_mode == "basnet":
        return BASNetLike(
            encoder_name=cfg.encoder_name,
            encoder_weights=encoder_weights,
            use_refinement=cfg.use_refinement,
            use_transformer=cfg.use_transformer,
        )

    raise ValueError(cfg.boundary_mode)