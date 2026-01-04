import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.flatten(2)
        targets = targets.flatten(2)

        intersection = (probs * targets).sum(-1)
        union = probs.sum(-1) + targets.sum(-1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()