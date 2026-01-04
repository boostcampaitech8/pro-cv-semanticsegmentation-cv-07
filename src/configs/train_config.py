from dataclasses import dataclass

@dataclass
class TrainConfig:
    # ===== 필수 인자 (non-default) =====
    model_name: str
    encoder_name: str
    save_name: str
    pretrained: str # "scratch" | "imagenet" | "radimagenet"

    seed: int

    batch_size: int
    num_workers_train: int
    num_workers_val: int

    lr: float
    num_epochs: int
    val_every: int
    num_patience: int

    use_wandb: bool

    # ===== 선택 인자 (default) =====
    boundary_mode: str = "none"
    use_refinement: bool = False
    use_transformer: bool = False
    boundary_detach: bool = True

    loss_mode: str = "bce"  # "bce" | "bce_dice" | "bce_dice_jaccard"