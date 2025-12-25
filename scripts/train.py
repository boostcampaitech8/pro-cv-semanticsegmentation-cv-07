import sys
import os
from src.data.train_data import XRayDataset
from src.configs.config import BATCH_SIZE, CLASSES, LR, SAVED_DIR, RANDOM_SEED, NUM_EPOCHS, VAL_EVERY
from src.utils.set_seed import set_seed
from src.engine.trainer import train
import os
import wandb
from dotenv import load_dotenv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import sys
import ml_collections
import torch

def main():
    set_seed(RANDOM_SEED)
    
    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)
    
    save_file_name = "SWIN-UNET_best_model.pt"
    checkpoint_path = os.path.join(SAVED_DIR, save_file_name)
    try:
        from external.Swin_Unet.networks.vision_transformer import SwinUnet 
        from external.Swin_Unet.config import get_config
        print("✅ Swin-Unet 모듈 로드 성공.")
        
    except ImportError as e:
        print(f"❌ SWIN-UNET 모듈 로드 실패: {e}")
    load_dotenv()
    
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=save_file_name,
        config={
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "random_seed": RANDOM_SEED,
            "num_epochs": NUM_EPOCHS,
            "val_every": VAL_EVERY,
        },
        settings=wandb.Settings(_disable_stats=False),
    )

    wandb.config.update({"monitor_memory": True})
    
    train_dataset = XRayDataset(is_train=True)
    valid_dataset = XRayDataset(is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    from external.Swin_Unet.config import _C
    import copy

    config = copy.deepcopy(_C)
    config.merge_from_file(
        "external/Swin_Unet/configs/swin_tiny_patch4_window7_224_lite.yaml"
    )

    

    config.MODEL.NUM_CLASSES = len(CLASSES)
    
    #512에 맞게끔 윈도우 사이즈?
    config.DATA.NUM_CLASSES = len(CLASSES)
    config.MODEL.SWIN.WINDOW_SIZE = 8

    config.DATA.IMG_SIZE = 512
    

    
    
    
    
    model = SwinUnet(config,num_classes=len(CLASSES))
    model = model.cuda()
    #사전학습된 가중치가 224 7 인데 가져오는게 맞을까?
    PRETRAINED_WEIGHTS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "external/Swin_Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
    )

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)
    start_epoch=40 #임의로 설정 . 왜냐면 model 저장할 떄 epoch를 저장 안해서. 40번 반복했잖아.
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint_model = torch.load(checkpoint_path, map_location='cuda')
        
        if isinstance(checkpoint_model, torch.nn.Module):
            model = checkpoint_model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
            print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch + 1}")




    





    train(model, train_loader, valid_loader, criterion, optimizer, save_file_name=save_file_name,start_epoch=start_epoch)


if __name__ == '__main__':
    main()