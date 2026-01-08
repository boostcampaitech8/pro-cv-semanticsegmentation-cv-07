
from external.Swin_Unet.config import _C
import os
import torch
import copy


def build_swin_unet(cfg,PRETRAINED_WEIGHTS_PATH=None,classes=29):
    try:
        from external.Swin_Unet.networks.vision_transformer import SwinUnet 
        from external.Swin_Unet.config import get_config
        print("Swin-Unet 모듈 로드 성공.")
        
    except ImportError as e:
        print(f"SWIN-UNET 모듈 로드 실패: {e}")
  

    
    config = copy.deepcopy(_C)
    config.merge_from_file(
        "external/Swin_Unet/configs/swin_tiny_patch4_window7_224_lite.yaml"
    )
    

    config.DATA.IMG_SIZE = cfg.img_size
    config.MODEL.SWIN.WINDOW_SIZE =cfg.window_size
   
    
        
        
        
    
    model = SwinUnet(config,num_classes=classes)

    if PRETRAINED_WEIGHTS_PATH is not None:
        if os.path.exists(PRETRAINED_WEIGHTS_PATH):
            print(f"✅ Loading pretrained weights from: {PRETRAINED_WEIGHTS_PATH}")
            
            # 가중치 파일 로드
            checkpoint = torch.load(PRETRAINED_WEIGHTS_PATH, map_location='cpu')
            
            # Swin-Unet 깃허브의 가중치 키는 'model' 아래에 저장되어 있습니다.
            state_dict = checkpoint.get('model', checkpoint) 
            
            # 최종 분류 레이어는 클래스 수가 다르므로 로드하지 않도록 strict=False를 사용합니다.
            model.load_state_dict(state_dict, strict=False)
            print("✅ Pretrained weights loaded successfully (skipping mismatching layers).")
        else:
            print(f"❌ Pretrained weights file not found at {PRETRAINED_WEIGHTS_PATH}. Starting training from scratch.")
    
    return model