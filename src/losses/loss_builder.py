import segmentation_models_pytorch as smp
import torch.nn as nn


def build_loss(cfg):
    
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