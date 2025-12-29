import segmentation_models_pytorch as smp


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

    elif cfg.pretrained == "radimagenet":
        encoder_weights = None  # 일단 None로 생성
    else:
        encoder_weights = None

    model = model_cls(
        encoder_name=cfg.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        decoder_use_batchnorm=False
    )

    # 🔹 RADImageNet weight 수동 로딩
    if cfg.pretrained == "radimagenet":
        ckpt_path = "/path/to/RadImageNet-ResNet50.pth"
        state = torch.load(ckpt_path, map_location="cpu")

        model.encoder.load_state_dict(state, strict=False)

    return model