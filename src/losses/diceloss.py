import torch
import torch.nn as nn

class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-5):
            super().__init__()
            self.smooth = smooth

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            probs = probs.reshape(probs.size(0), -1)
            targets = targets.reshape(targets.size(0), -1)
            intersection = (probs * targets).sum(1)
            dice = (2 * intersection + self.smooth) / (
                probs.sum(1) + targets.sum(1) + self.smooth
            )
            return 1 - dice.mean()
