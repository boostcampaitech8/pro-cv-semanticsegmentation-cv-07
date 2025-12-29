import torch

def _strip_prefix(state_dict, prefixes=("module.", "model.", "encoder.")):
    """체크포인트에 붙은 prefix를 제거해서 encoder에 맞추기."""
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        out[new_k] = v
    return out

def load_encoder_weights(encoder, pretrained: str, encoder_name: str, ckpt_paths: dict):
    """
    encoder: smp encoder object (smp.encoders.get_encoder(...) 로 만든 것)
    pretrained: scratch|imagenet|radimagenet|torchxrayvision
    encoder_name: 'resnet50' etc. (필요하면 분기)
    ckpt_paths: {"radimagenet": "...pth", "torchxrayvision": "...pth"} 처럼 전달
    """
    if pretrained in ["scratch", "imagenet"]:
        return  # imagenet은 생성할 때 weights="imagenet"로 처리, scratch는 아무것도 안 함

    if pretrained == "radimagenet":
        path = ckpt_paths.get("radimagenet")
        if path is None:
            raise ValueError("radimagenet ckpt path is None")

        ckpt = torch.load(path, map_location="cpu")
        # ckpt가 state_dict 자체일 수도, {"state_dict": ...}일 수도
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = _strip_prefix(state)

        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[RADImageNet] missing={len(missing)} unexpected={len(unexpected)}")
        return

    if pretrained == "torchxrayvision":
        path = ckpt_paths.get("torchxrayvision")
        if path is None:
            raise ValueError("torchxrayvision ckpt path is None")

        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = _strip_prefix(state)

        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[TorchXRayVision] missing={len(missing)} unexpected={len(unexpected)}")
        return

    raise ValueError(f"Unknown pretrained: {pretrained}")
