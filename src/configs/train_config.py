from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str
    encoder_name: str
    save_name: str
    loss_type: str
    scheduler: str
    optimizer: str
    
    seed: int

    batch_size: int
    num_workers_train: int
    num_workers_val: int

    lr: float
    num_epochs: int
    val_every: int
    num_patience: int

    use_wandb: bool