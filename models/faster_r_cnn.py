
"""
Faster R-CNN Implementation
"""

import torch

class FasterRCNN(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, rpn: torch.nn.Module, roi_heads: torch.nn.Module, transform: torch.nn.Module):
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self,):
        pass


