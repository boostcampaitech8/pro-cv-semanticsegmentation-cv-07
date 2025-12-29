import segmentation_models_pytorch as smp
import torch


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


def build_smp_model(cfg, in_channels=3, classes=29):
    model_name = cfg.model_name.lower()
    if model_name not in SMP_MODELS:
        raise ValueError(f"Unknown model_name {cfg.model_name}")

    model_cls = SMP_MODELS[model_name]
    # 🔹 encoder weight 선택
    if cfg.pretrained == "imagenet":
        encoder_weights = "imagenet"
    elif cfg.pretrained == "scratch":
        encoder_weights = None
    elif cfg.pretrained == "radimagenet":
        encoder_weights = None
    else:
        raise ValueError(f"Unknown pretrained option: {cfg.pretrained}")


    model = model_cls(
        encoder_name=cfg.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        decoder_use_batchnorm=False
    )

    print(
        f"[Encoder Init] "
        f"model={cfg.model_name}, "
        f"encoder={cfg.encoder_name}, "
        f"pretrained={cfg.pretrained}, "
        f"encoder_weights={encoder_weights}"
    )

    # 🔹 RADImageNet weight 수동 로딩
    if cfg.pretrained == "radimagenet":
        ckpt_path = "/path/to/RadImageNet-ResNet50.pth"
        state = torch.load(ckpt_path, map_location="cpu")

        model.encoder.load_state_dict(state, strict=False)

    return model