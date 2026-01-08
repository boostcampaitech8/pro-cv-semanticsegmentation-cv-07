import yaml
#import copy
from dataclasses import dataclass#, field, asdict
from typing import Any, Dict


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
    model_conf: str # for HRNet
    encoder_name: str
    window_size:str
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
    
    poly_power: float

    boundary_mode: str
    use_refinement: bool
    use_transformer: bool
    boundary_detach: bool
    pretrained: str
