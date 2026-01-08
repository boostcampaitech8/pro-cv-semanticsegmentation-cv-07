from src.configs.run_config import parse_args, build_config
from src.data.train_data import XRayDataset
from src.utils.set_seed import set_seed
from src.engine.trainer import train
from src.models.smp_model import build_smp_model
from src.losses.loss_builder import build_loss
from src.models.scheduler import build_scheduler
from src.models.optimizer import get_optimizer
import os
import wandb
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader


def main():
    args = parse_args()
    cfg = build_config(args)
    
    set_seed(cfg.seed)
    
    if not os.path.exists(cfg.saved_root):                                                           
        os.makedirs(cfg.saved_root)
    
    if cfg.use_wandb:
        load_dotenv()
    
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"{cfg.saved_name}",
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
    
    train_dataset = XRayDataset(cfg, is_train=True)
    if cfg.total:
        valid_loader = None
    else:
        valid_dataset = XRayDataset(cfg, is_train=False)
    
        valid_loader = DataLoader(
            dataset=valid_dataset, 
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers_val,
            drop_last=False
        )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers_train,
        drop_last=True,
    )
    
    model = build_smp_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = build_loss(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    train(model, train_loader, valid_loader, criterion, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()