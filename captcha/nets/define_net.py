from typing import Optional

import torch
import torchvision.models as models
from torch import nn

from captcha.config import torch_config


def define_net(
    freeze_grads: bool = False,
    outputs: int = 1,
    pretrained: bool = False,
    weights: Optional[str] = None,
):
    if (weights is not None) and weights:
        model = torch.load(weights, map_location="cpu")
    else:
        pretrained_weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=pretrained_weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if freeze_grads:
            for params in model.parameters():
                params.requires_grad_ = False

        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=outputs, bias=True
        )

    model.to(torch_config.device)

    return model
