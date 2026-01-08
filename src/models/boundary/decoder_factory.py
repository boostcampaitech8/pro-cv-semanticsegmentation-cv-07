import inspect
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

"""
    SMP(UnetDecoder) 버전 차이를 흡수하기 위한 factory.
    - use_batchnorm / decoder_use_batchnorm
    - center / attention 옵션 존재 여부 차이 대응
    """
def build_unet_decoder(
    encoder_channels,
    decoder_channels,
    n_blocks=5,
    use_bn=False,
    center=False,
):
    # SMP 버전에 따라 파라미터 이름이 다르므로
    # signature inspection으로 안전하게 생성
    sig = inspect.signature(UnetDecoder.__init__)
    params = sig.parameters

    kwargs = dict(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        n_blocks=n_blocks,
    )

    # 🔹 batchnorm 옵션
    if "use_batchnorm" in params:
        kwargs["use_batchnorm"] = use_bn
    elif "decoder_use_batchnorm" in params:
        kwargs["decoder_use_batchnorm"] = use_bn

    # 🔹 center 옵션 (❗ 여기 추가)
    if "center" in params:
        kwargs["center"] = center

    # 🔹 attention 옵션
    if "attention_type" in params:
        kwargs["attention_type"] = None

    return UnetDecoder(**kwargs)