from src.configs.defaults import SAVED_DIR, CLASSES
from src.configs.run_config import parse_args, build_config
from src.data.train_data import XRayDataset
from src.utils.set_seed import set_seed
from src.engine.trainer import train
from src.models.smp_model import build_smp_model
import os
import wandb
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main():
    args = parse_args()
    cfg = build_config(args)
    
    set_seed(cfg.seed)
    
    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)
    
    if cfg.use_wandb:
        load_dotenv()
    
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"{cfg.model_name}({cfg.encoder_name})",
            config={
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "random_seed": cfg.seed,
                "num_epochs": cfg.num_epochs,
                "val_every": cfg.val_every,
            },
            settings=wandb.Settings(_disable_stats=False),
        )

        wandb.config.update({"monitor_memory": True})
    
    train_dataset = XRayDataset(is_train=True, split_file="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/src/datasets/splits/fold_0_train.txt")
    valid_dataset = XRayDataset(is_train=False, split_file="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/src/datasets/splits/fold_0_val.txt")
    
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
    
    model = build_smp_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=1e-6)

    train(model, train_loader, valid_loader, criterion, optimizer, cfg)


if __name__ == '__main__':
    main()