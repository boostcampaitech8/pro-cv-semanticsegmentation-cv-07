import torch
from torch import nn
import torch.nn.functional as F
from src.losses.bce_dice import DiceLoss

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal loss on top of BCEWithLogits for multi-label segmentation.

    FL = alpha * (1 - pt)^gamma * BCE
      where pt = sigmoid(logit) if y=1 else (1 - sigmoid(logit))

    - alpha can be:
      * float: apply same alpha to positive class (neg uses 1-alpha)
      * None: no alpha balancing
      * Tensor shape [C]: per-class alpha for positive class
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        # element-wise BCE (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,C,H,W]

        # pt
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)  # [B,C,H,W]

        focal_factor = (1.0 - pt).pow(self.gamma)

        loss = focal_factor * bce

        # alpha balancing (optional)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_pos = float(self.alpha)
                alpha_t = alpha_pos * targets + (1.0 - alpha_pos) * (1.0 - targets)
                loss = alpha_t * loss
            else:
                # assume Tensor [C]
                alpha = self.alpha.to(logits.device).view(1, -1, 1, 1)
                alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
                loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalBCEDiceLoss(nn.Module):
    """
    Your requested: Focal(BCE term only) + Dice.

    total = w * FocalBCE + (1-w) * Dice
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        gamma: float = 2.0,
        alpha=None,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.w = bce_weight
        self.focal_bce = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha, reduction="mean")
        self.dice = DiceLoss(smooth=dice_smooth)

    def compute_parts(self, logits, targets):
        loss_focal = self.focal_bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_total = self.w * loss_focal + (1.0 - self.w) * loss_dice
        return loss_total, loss_focal, loss_dice

    def forward(self, logits, targets):
        loss_total, _, _ = self.compute_parts(logits, targets)
        return loss_total
