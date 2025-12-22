from src.data.train_data import XRayDataset
from src.configs.config import BATCH_SIZE, CLASSES, LR, SAVED_DIR, RANDOM_SEED, NUM_EPOCHS, VAL_EVERY, NUM_PATIENCE
from src.utils.set_seed import set_seed
from src.engine.trainer import train
import os
import wandb
from dotenv import load_dotenv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


def main():
    set_seed(RANDOM_SEED)
    
    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)
    
    save_file_name = 'unet_baseline_best_model.pt'
    
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
    
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=29,
    )
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

    train(model, train_loader, valid_loader, criterion, optimizer, save_file_name=save_file_name, num_patience=NUM_PATIENCE)


if __name__ == '__main__':
    main()