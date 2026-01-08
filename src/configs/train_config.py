from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str
    saved_root: str
    saved_name: str
    split_file_root: str
    
    total:bool
    tta: bool  
    
    img_size: int
    use_scale: bool
    use_rotate: bool
    use_flip: bool
    use_contrast: bool
    
    model_name: str
    encoder_name: str
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

    # ===== boundary options (ADD) =====
    boundary_mode: str = "none"
    use_refinement: bool = False
    use_transformer: bool = False
    boundary_detach: bool = True
    loss_mode: str = "BD"