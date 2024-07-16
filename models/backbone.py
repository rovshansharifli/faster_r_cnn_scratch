
"""
ResNet Implementation

Based on "Deep Residual Learning for Image Recognition" paper
https://arxiv.org/pdf/1512.03385
"""

import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, Callable, Type


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

        # This is the residual connection (or simply the shortcut)
        out += identity

        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, # A function (or callable) that accepts any number of arguments and returns an nn.Module
                 groups: int = 1,
                 width_per_group: int = 64, 
                 zero_init_residual: bool = False,
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

        self.groups = groups
        self.base_width = width_per_group

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
        
        # (1x1, 64; 3x3, 64; 1x1, 256) x 3 
        # (1x1, 128; 3x3, 128; 1x1, 512) x 4 
        # (1x1, 256; 3x3, 256; 1x1, 1024) x 6 
        # (1x1, 512; 3x3, 512; 1x1, 2048) x 3 
        layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Classification head
        self.fc = nn.Linear(512 * 4, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                # elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                #     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self, 
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = True
        ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != (planes * 4):
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes, 
                    out_channels=planes * 4,   # last layer is 4 times bigger 
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                norm_layer(planes * 4),
            )

        layers = []
        # As mentioned on ResNet paper:
        # Downsampling is performed by conv3 1, conv4 1, 
        # and conv5 1 with a stride of 2
        layers.append(
            Bottleneck(in_planes=self.inplanes, 
                       planes=planes,
                       stride=stride,
                       downsample=downsample,
                       groups=self.groups,
                       base_width=self.base_width,
                       dilation=previous_dilation,
                       norm_layer=norm_layer),
            )
        
        self.inplanes = planes * 4

        # Adding the rest of the layer
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    in_planes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
