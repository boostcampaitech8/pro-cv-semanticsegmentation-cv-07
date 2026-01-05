import torch.nn as nn
from .dice_loss import DiceLoss
from .jaccard_loss import JaccardLoss

class CombinedLoss(nn.Module):
    def __init__(self, mode="bce"):
        super().__init__()
        self.mode = mode

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.jaccard = JaccardLoss()

    def forward(self, logits, targets):
        if self.mode == "bce":
            return self.bce(logits, targets)

        elif self.mode == "bce_dice":
            return self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

        elif self.mode == "bce_dice_jaccard":
            return (
                self.bce(logits, targets)
                + self.dice(logits, targets)
                + self.jaccard(logits, targets)
            )

        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")