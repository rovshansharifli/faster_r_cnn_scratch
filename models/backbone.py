
"""
ResNet Implementation

Based on "Deep Residual Learning for Image Recognition" paper
https://arxiv.org/pdf/1512.03385
"""

import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, Callable


class Bottleneck(nn.Module):
    """
    So on the original paper the bottleneck places the stride 
    for downsampling at 1x1 (conv1) but torchvision does
    it on 3x3 (conv2) as it improves the accuracy

    We will follow the torchvision implementation so that
    there will be no problem on loading the pretrained
    weights on torchvision
    """

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            base_width: int = 64,
            groups: int = 1,
            dilation: int = 1,
            downsample: Optional[nn.Module] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # Bottleneck architecture will contain:
        # (1x1, 64; 3x3, 64; 1x1, 256) x 3 
        # (1x1, 128; 3x3, 128; 1x1, 512) x 4 
        # (1x1, 256; 3x3, 256; 1x1, 1024) x 6 
        # (1x1, 512; 3x3, 512; 1x1, 2048) x 3 

        # 1x1 conv and batch norm
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, 
            out_channels=width, 
            kernel_size=1, 
            stride=1, 
            bias=False)
        self.bn1 = norm_layer(width)

        # 3x3 conv and batch norm
        self.conv2 = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
            )
        self.bn2 = norm_layer(width)

        # 1x1 conv and batch norm
        self.conv3 = nn.Conv2d(
            in_channels=width, 
            out_channels=planes * 4,   # last layer is 4 times bigger 
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Forward pass of 1x1 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Forward pass of 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Forward pass of 1x1
        out = self.conv3(out)
        out = self.bn3(out)

        # As mentioned on ResNet paper:
        # We perform downsampling directly by
        # convolutional layers that have a stride of 2
        # Downsampling is performed by conv3 1, conv4 1, 
        # and conv5 1 with a stride of 2
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, # A function (or callable) that accepts any number of arguments and returns an nn.Module
                 
                 ):
        # Initializing the attributes on the parent Modules class
        super().__init__()

        # Paper indicates batch normalization is applied after
        # each convolutional and before activation as mentioned
        # on paper: "Batch normalization: Accelerating deep network
        # training by reducing internal covariate shift"
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Minimum number of filters;
        # it starts from 64, 128, 256, 512 etc.
        self.inplanes = 64

        # Dilation is a parameter that controls the spacing 
        # between the kernel elements
        self.dilation = 1

        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError(
        #         "replace_stride_with_dilation should be None "
        #         f"or a 3-element tuple, got {replace_stride_with_dilation}"
        #     )

        # self.groups = groups
        # self.base_width = width_per_group

        # ResNet50 Architecture -> as on ResNet paper mentioned
        
        # 1 - General part for ResNet architecture
        # 7x7, 64, stride 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # batch norm after convolutional layer
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 3x3 max pool, stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 2 - ResNet50 specific part
        # self.block = Bottleneck


        # self.layer1 = self._make_layer(Bottleneck, 64, layers[0])

        # _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
        








