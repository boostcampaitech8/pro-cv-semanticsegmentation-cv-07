import torch
from torch import nn
import torch.nn.functional as F


class HRNetOcrLossWrapper(nn.Module):
    """
    Wrap any base criterion to handle HRNet/MMseg outputs:
    - outputs can be Tensor or list/tuple of Tensors (aux..., main)
    - will interpolate to targets size
    - apply aux_weight to all but last, main_weight to last
    """
    def __init__(
        self,
        base_criterion: nn.Module,
        aux_weight: float = 0.4,
        main_weight: float = 1.0,
        return_dict=False
    ):
        super().__init__()
        self.base = base_criterion
        self.aux_weight = aux_weight
        self.main_weight = main_weight
        self.return_dict = return_dict

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # print('outputs.shape in wrapper', outputs.shape)
        def _compute(out):
            # base에 compute_parts가 있으면 breakdown 사용
            if hasattr(self.base, "compute_parts"):
                return self.base.compute_parts(out, targets)  # (total, focal, dice)
            else:
                total = self.base(out, targets)
                return total, None, None
            
        if isinstance(outputs, (list, tuple)):
            total = 0.0
            focal_sum = 0.0
            dice_sum = 0.0
            focal_cnt = 0
            dice_cnt = 0

            for i, out in enumerate(outputs):
                if out.shape[-2:] != targets.shape[-2:]:
                    out = F.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)

                w = self.main_weight if i == (len(outputs) - 1) else self.aux_weight
                loss_total, loss_focal, loss_dice = _compute(out)

                total = total + w * loss_total
                if loss_focal is not None:
                    focal_sum = focal_sum + w * loss_focal
                    focal_cnt += 1
                if loss_dice is not None:
                    dice_sum = dice_sum + w * loss_dice
                    dice_cnt += 1

            if not self.return_dict:
                return total

            log_dict = {"loss/total": total.detach()}
            if focal_cnt > 0:
                log_dict["loss/focal_bce"] = focal_sum.detach()
            if dice_cnt > 0:
                log_dict["loss/dice"] = dice_sum.detach()
            return total, log_dict

        else:
            out = outputs
            if out.shape[-2:] != targets.shape[-2:]:
                out = F.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)

            loss_total, loss_focal, loss_dice = _compute(out)

            if not self.return_dict:
                return loss_total

            log_dict = {"loss/total": loss_total.detach()}
            if loss_focal is not None:
                log_dict["loss/focal_bce"] = loss_focal.detach()
            if loss_dice is not None:
                log_dict["loss/dice"] = loss_dice.detach()
            return loss_total, log_dict
