import torch
import torch.nn as nn
from torch import Tensor

from typing import List

from typing import List, Tuple

import torch
from torch import Tensor

# Taken from torchvision as is
class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class AnchorGenerator(nn.Module):
    def __init__(self, ):
        # On paper the size and aspect ratios are mentioned:
        # For anchors, we use 3 scales with box areas of 128,
        # 256, and 512 pixels and 3 aspect ratios of 1:1, 1:2,
        # and 2:1.
        # On torchvision implementation however, sizes are
        # ((32,), (64,), (128,), (256,), (512,))
        # and aspect ratios are
        # ((0.5, 1.0, 2.0),) * len(sizes)
        # for ResNet50 backbone.
        # This is probably dut to the backbone change. The original paper
        # is using ZF and VGG16 networks as backbone.
        super().__init__()
        self.scales = ((32,), (64,), (128,), (256,), (512,))
        self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.scales)

        self.cell_anchors = [] 
        for scales, aspect_ratios in zip(self.scales, self.aspect_ratios):

            scales = torch.as_tensor(scales, dtype=torch.float32, device=torch.device('cpu'))
            aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32, device=torch.device('cpu'))

            h_ratios = torch.sqrt(aspect_ratios)
            w_ratios = 1 / h_ratios

            # [:, None] -> this changes the rows to columns
            # Well techniquely view(-1) is also doing the same
            ws = (w_ratios.view(-1) * scales.view(-1)).view(-1)
            hs = (h_ratios.view(-1) * scales.view(-1)).view(-1)

            # As the anchors are zero-centered we have negative and 
            # positive values on base anchors
            base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

            self.cell_anchors.append(base_anchors.round())

    # Function is taken from torchvision as is
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:

        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors


    # Function is taken from torchvision as is
    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]

        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1])
            ] for g in grid_sizes
        ]
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


class RegionProposalNetwork(nn.Module):
    """
    Input should take any size of image and outputs a set of
    rectangular object proposals, each with an objectness score

    This process is implemented using a fully convolutional network

    To share computation with Fast R-CNN, both nets share a common 
    set of convolutional layers.
    """

    def __init__(self,):
        pass

        
