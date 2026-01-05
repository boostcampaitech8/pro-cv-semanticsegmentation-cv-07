from src.configs.defaults import SAVED_DIR, CLASSES
from src.configs.run_config import parse_args, build_config
from src.data.train_data import XRayDataset
from src.utils.set_seed import set_seed
from src.engine.trainer import train
from src.models.build_model import build_model
import os
import wandb
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.losses.combined_loss import CombinedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

def main():
    args = parse_args()
    cfg = build_config(args)
    
    set_seed(cfg.seed)
    
    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)
    
    if cfg.use_wandb:
        load_dotenv()
    
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        run_name = cfg.save_name.replace(".pt", "")

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,   # ✅ save_name 기반
            config={
                "model": cfg.model_name,
                "encoder": cfg.encoder_name,
                "pretrained": cfg.pretrained,
                "boundary_mode": cfg.boundary_mode,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "seed": cfg.seed,
                "num_epochs": cfg.num_epochs,
            },
        )

        wandb.config.update({"monitor_memory": True})
    
    FOLD = args.fold  # 바꾸면서 실험

    train_dataset = XRayDataset(fold=FOLD, is_train=True)
    valid_dataset = XRayDataset(fold=FOLD, is_train=False)

    print(f"[Dataset] Fold {FOLD} | Train size: {len(train_dataset)} | Val size: {len(valid_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers_train,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers_val,
        drop_last=False
    )
    
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    seg_criterion = CombinedLoss(mode=cfg.loss_mode)
    boundary_criterion = CombinedLoss(mode="bce")  # ← 고정
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",        # dice를 최대화할 거라서
        factor=0.5,
        patience=5,
        threshold=1e-3,
        min_lr=1e-6,
        verbose=True,
    )
    train(model, train_loader, valid_loader, seg_criterion, boundary_criterion, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()