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


def build_smp_model(model_name, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=29):
    
    model_name = model_name.lower()
    if model_name not in SMP_MODELS:
        raise ValueError(f"Unknown model_name {model_name}, choose from {list(SMP_MODELS.keys())}")

    model_cls = SMP_MODELS[model_name]
    
    if model_name in ["segformer", "dpt", "upernet"]:
        return model_cls(
            in_channels=in_channels,
            classes=classes,
        )
    else:
        return model_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )