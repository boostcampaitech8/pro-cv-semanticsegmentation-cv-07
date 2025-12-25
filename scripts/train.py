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
        batch_size=8,
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
    config.DATA.NUM_CLASSES = len(CLASSES)
    

    config.DATA.IMG_SIZE = 512

    # 🔹 Swin 설정
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.WINDOW_SIZE = 16
    
    
    
    model = SwinUnet(config,num_classes=len(CLASSES))
    model = model.cuda()
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    



    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)





#    train(model, train_loader, valid_loader, criterion, optimizer, save_file_name=save_file_name)


if __name__ == '__main__':
    main()