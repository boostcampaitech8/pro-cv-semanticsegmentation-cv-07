import segmentation_models_pytorch as smp
import torch.nn as nn
from src.losses.focal_bce import FocalBCEDiceLoss
from src.losses.wrappers import HRNetOcrLossWrapper
from src.configs.defaults import CLASSES
import torch


def build_loss(cfg):
    
    if cfg.model_name == 'hrnet' or cfg.loss_type == 'weighted_focal':
        # HRNet 학습 시 사용했던 설정
        # alpha 설정
        alpha = torch.full((len(CLASSES),), 0.25)
        expansion_group = ["Pisiform", "Trapezoid", "Hamate"]
        shrinkage_group = ["Trapezium", "Capitate", "Lunate", "Scaphoid", "Triquetrum", "finger-1"]
        for cls_name in expansion_group:
            if cls_name in CLASSES:
                alpha[CLASSES.index(cls_name)] = 0.60
                
        for cls_name in shrinkage_group:
            if cls_name in CLASSES:
                alpha[CLASSES.index(cls_name)] = 0.85

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        alpha = alpha.to(device)

        base_criterion = FocalBCEDiceLoss(
            bce_weight=0.5,
            gamma=2.0,
            alpha=alpha,       
            dice_smooth=1.0,
        )
        
        if cfg.model_name == 'hrnet':
            return HRNetOcrLossWrapper(base_criterion, aux_weight=0.4, main_weight=1.0, return_dict=True)
        else:
            return base_criterion

    losses = []

    if "B" in cfg.loss_type:
        losses.append(nn.BCEWithLogitsLoss())

    if "D" in cfg.loss_type:
        losses.append(smp.losses.DiceLoss(mode="multilabel"))

    if "J" in cfg.loss_type:
        losses.append(smp.losses.JaccardLoss(mode="multilabel"))

    if "F" in cfg.loss_type:
        losses.append(smp.losses.FocalLoss(mode="multilabel"))

    if "T" in cfg.loss_type:
        losses.append(smp.losses.TverskyLoss(mode="multilabel"))

    assert len(losses) > 0, f"Invalid cfg.loss: {cfg.loss_type}"

    def criterion(logits, targets):
        total = 0.0
        for loss_fn in losses:
            total += loss_fn(logits, targets)
        return total

    return criterion