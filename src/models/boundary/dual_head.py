import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadBoundaryNet(nn.Module):
    """
    Dual-head 구조:
    - segmentation head: 주 task
    - boundary head: 보조 task

    detach_boundary 옵션:
    - True  : boundary loss가 decoder feature에 역전파되지 않음
    - False : boundary가 decoder 학습에 직접 관여
    """
    def __init__(self, base_model, detach_boundary=True):
        super().__init__()
        self.base_model = base_model
        self.detach_boundary = detach_boundary

        # base SMP model 구성요소 재사용
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.seg_head = base_model.segmentation_head

        # decoder 출력 채널
        decoder_out_ch = self.decoder.out_channels[-1]

        # boundary head만 추가
        self.boundary_head = nn.Conv2d(
            decoder_out_ch, 1, kernel_size=1
        )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        seg_logits = self.seg_head(decoder_output)
        
        # detach 목적:
        # boundary loss가 shared decoder를 과도하게 지배하는 것을 방지
        boundary_feat = (
            decoder_output.detach()
            if self.detach_boundary else decoder_output
        )
        boundary_logits = self.boundary_head(boundary_feat)

        return {
            "seg": seg_logits,
            "boundary": boundary_logits,
        }
