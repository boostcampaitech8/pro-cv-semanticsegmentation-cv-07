import segmentation_models_pytorch as smp
from src.models.boundary.dual_head import DualHeadBoundaryNet
from src.models.boundary.basnet_like import BASNetLike


SMP_MODELS = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "upernet": smp.UPerNet,
    "segformer": smp.Segformer,
    "dpt": smp.DPT,
}


def build_smp_model(cfg, encoder_weights="imagenet", in_channels=3, classes=29):
    
    model_name = cfg.model_name.lower()
    

    if model_name == "swin_unet":
        from src.models.swin_unet import build_swin_unet
        return build_swin_unet(cfg)
      
    elif model_name == 'hrnet':
        from src.models.hrnet_mmseg import get_mmseg_model
        return get_mmseg_model(cfg.model_conf, classes)

    elif model_name not in SMP_MODELS:
        raise ValueError(f"Unknown model_name {cfg.model_name}, choose from {list(SMP_MODELS.keys())}")
    
    # encoder_weights 결정

    if cfg.pretrained == "imagenet":
        encoder_weights = "imagenet"
    elif cfg.pretrained == "scratch":
        encoder_weights = None
    else:
        raise ValueError(f"Unknown pretrained: {cfg.pretrained}")

    model_cls = SMP_MODELS[model_name]
    
    base_model = model_cls(
        encoder_name=cfg.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None, 
        decoder_use_batchnorm=False
    )


    # boundary 분기

    if cfg.boundary_mode == "none":
        return base_model

    if cfg.boundary_mode == "dual":
        return DualHeadBoundaryNet(
            base_model=base_model,
            detach_boundary=cfg.boundary_detach,
        )

    if cfg.boundary_mode == "basnet":
        return BASNetLike(
            base_model=base_model,
            use_refinement=cfg.use_refinement,
            use_transformer=cfg.use_transformer,
        )

    raise ValueError(f"Unknown boundary_mode: {cfg.boundary_mode}")