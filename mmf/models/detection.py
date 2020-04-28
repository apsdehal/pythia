from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FastRCNNPredictor,
    TwoMLPHead,
)
from torchvision.models.resnet import _resnet
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.poolers import initLevelMapper

from mmf.modules.detection._mb_anchor_generator import MBAnchorGenerator


class CustomTwoMLPHead(TwoMLPHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clear(self):
        self._fc6 = None
        self._fc7 = None

    def data(self):
        return {"fc6": self._fc6, "fc7": self._fc7}

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        self._fc6 = x.clone().detach()
        x = F.relu(self.fc7(x))
        self._fc7 = x.clone().detach()

        return x


class CustomFastRCNNPredictor(FastRCNNPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clear(self):
        self._scores = None
        self._bbox_deltas = None

    def data(self):
        return {"scores": self._scores, "bbox_deltas": self._bbox_deltas}

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        self._scores = scores.clone().detach()
        self._bbox_deltas = bbox_deltas.clone().detach()
        return scores, bbox_deltas


class CustomMBFasterRCNN(FasterRCNN):
    def __init__(
        self,
        backbone,
        num_classes=1601,
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        representation_size=2048,
        *args,
        **kwargs
    ):
        out_channels = backbone.out_channels
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )
            scales = [0.25, 0.125, 0.0625, 0.03125]
            lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
            lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
            box_roi_pool.scales = scales
            box_roi_pool.map_levels = initLevelMapper(int(lvl_min), int(lvl_max))

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = CustomTwoMLPHead(
                out_channels * resolution ** 2, representation_size
            )

        if box_predictor is None:
            box_predictor = CustomFastRCNNPredictor(representation_size, num_classes)

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = (0.5, 1.0, 2.0)

        anchor_generator = MBAnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            anchor_strides=(4, 8, 16, 32, 64),
        )
        super().__init__(
            backbone,
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            rpn_anchor_generator=anchor_generator,
        )

        self.data_sources = [box_head, box_predictor]

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        for source in self.data_sources:
            for item, val in source.data().items():
                start_index = 0
                for idx in range(len(proposals)):
                    end_index = start_index + proposals[idx].size(0)
                    detections[idx][item] = val[start_index:end_index]
                    start_index = end_index
            source.clear()

        for idx, v in enumerate(proposals):
            detections[idx]["proposals"] = v

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)


def build_custom_mb_faster_rcnn():
    backbone = _resnet(
        "custom_mb",
        torchvision.models.resnet.Bottleneck,
        [3, 8, 36, 3],
        False,
        False,
        groups=32,
        width_per_group=8,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    )
    for name, parameter in backbone.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name:
            parameter.requires_grad_(False)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 512
    with_fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    return CustomMBFasterRCNN(with_fpn, num_classes=1601)
