
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

        # So Faster R-CNN has 2 modules
        # 1. Deep Fully Convolutional network -> to propose regions
        # 2. Fast R-CNN detector that uses the proposed regions

        # So this region proposal is a simple 'attention' mechanism
        # the RPN module tell the Fast R-CNN module where to look

        # 1. RPN -> implemented on rpn.py file



    def forward(self,):
        pass


