import torch
import torch.nn as nn

class ClassWiseLogitScaler(nn.Module):
    """
    Learnable class-wise logit scaling
    logit' = scale * logit + bias
    """
    def __init__(self, num_classes):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_classes))
        self.bias  = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        # logits: (B, C, H, W)
        scale = self.scale.view(1, -1, 1, 1)
        bias  = self.bias.view(1, -1, 1, 1)
        return logits * scale + bias