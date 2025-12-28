import sys
import os
from src.data.train_data import XRayDataset
from src.configs.config import BATCH_SIZE, CLASSES, LR, SAVED_DIR, RANDOM_SEED, NUM_EPOCHS, VAL_EVERY
from src.utils.set_seed import set_seed
from src.engine.trainer import train
from src.utils.earlystopping import EarlyStopping
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
from src.losses.diceloss import DiceLoss

# loss 바꿈. 이거 확인
#데이터 경로..
def main():
    set_seed(RANDOM_SEED)
    BATCH_SIZE=4
    NUM_EPOCHS=200
    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)
    
    save_file_name = "SWIN-UNET_best_model_1024_4.pt"
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
            "early_stopping_patience": 10
        },
        settings=wandb.Settings(_disable_stats=False),
    )

    wandb.config.update({"monitor_memory": True})
    
    FOLD = 0  # 바꾸면서 실험

    train_dataset = XRayDataset(fold=FOLD, is_train=True)
    valid_dataset = XRayDataset(fold=FOLD, is_train=False)

    print("train_dataset:",train_dataset.__len__())
    print("valid_dataset:",valid_dataset.__len__())
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

    config.DATA.IMG_SIZE = 1024
    
    config.NUM_EPOCHS = 200
    #이거 early stop 걸고 , 모델 계속돌리자.
    
    
    
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
    
    
    criterion = lambda x, y: (
        nn.BCEWithLogitsLoss()(x, y.float()) +
        DiceLoss()(x, y)
    )

    #criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)
    start_epoch=0 #임의로 설정 . 왜냐면 model 저장할 떄 epoch를 저장 안해서. 40번 반복했잖아.
    
    
    early_stopping = EarlyStopping(
        patience=10,  # 15 epoch 동안 개선이 없으면 중단
        verbose=True,
        path=checkpoint_path
    )
    
    
    # if os.path.exists(checkpoint_path):
    #     print(f"🔄 Loading checkpoint from {checkpoint_path}")
    #     try:
    #         # 1. 일단 로드합니다.
    #         loaded_data = torch.load(checkpoint_path, map_location=device)
            
    #         # 2. 로드된 데이터가 "모델 객체(옛날 방식)"인지 확인합니다.
    #         if isinstance(loaded_data, nn.Module):
    #             print("⚠️ Old checkpoint format detected (Model Object).")
                
    #             # 모델 가중치만 추출해서 현재 모델에 덮어씌움
    #             model.load_state_dict(loaded_data.state_dict())
                
    #             #용
                
    #         # 3. 로드된 데이터가 "딕셔너리(새 방식)"인지 확인합니다.
    #         elif isinstance(loaded_data, dict):
    #             print("✅ New checkpoint format detected (Dictionary).")
                
    #             model.load_state_dict(loaded_data['model_state_dict'])
                
    #             if 'optimizer_state_dict' in loaded_data:
    #                 optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
                
    #             start_epoch = loaded_data.get('epoch', 0)
                
    #         else:
    #             print("❌ Unknown checkpoint format.")

    #         print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
            
    #     except Exception as e:
    #         print(f"❌ Checkpoint load failed ({e}). Starting training from scratch.")
    # else:
    #     print("❌ Checkpoint not found. Starting training from scratch.")


    



    train(model, train_loader, valid_loader, criterion, optimizer, save_file_name=save_file_name,start_epoch=start_epoch,early_stopping=early_stopping)


if __name__ == '__main__':
    main()