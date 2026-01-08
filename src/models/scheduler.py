import torch.optim as optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def build_scheduler(cfg, optimizer, steps_per_epoch=None):
    if cfg.scheduler is None or cfg.scheduler == "none":
        return None

    elif cfg.scheduler == "warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg.num_epochs,
            cycle_mult=1.0,
            max_lr=cfg.lr,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=1.0
        )

    elif cfg.scheduler == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

    elif cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=1e-6
        )

    elif cfg.scheduler == "poly":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for poly scheduler")
        
        total_iters = steps_per_epoch * cfg.num_epochs
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_iters,
            power=cfg.poly_power
        )

    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler}")

    return scheduler