from typing import Optional

import torch
import torchvision.models as models
from torch import nn

from captcha.config import torch_config


def define_net(
    freeze_grads: bool = False,
    outputs: int = 1,
    pretrained: bool = False,
    weights: Optional = None,
):
    pretrained_weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=pretrained_weights)

    if freeze_grads:
        for params in model.parameters():
            params.requires_grad_ = False

    model.fc = nn.Linear(
        in_features=model.fc.in_features, out_features=outputs, bias=True
    )

    if weights is not None:
        model = torch.load(weights, map_location="cpu")

    model.to(torch_config.device)

    return model
