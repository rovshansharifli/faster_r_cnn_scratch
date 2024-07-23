from models.backbone import ResNet50
import torch

# model = ResNet50()
# x = torch.rand([1, 3, 300, 300])

# print(model.forward(x).size())

# from torchvision.models import resnet50
# model = resnet50()
# print(model.forward(x))

# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

from models.rpn import AnchorGenerator

AnchorGenerator()


