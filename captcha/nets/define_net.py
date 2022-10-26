from typing import Optional

import torch
import torchvision.models as models
from torch import nn

from captcha.config import torch_config


def define_net(
    model_name: str = "resnet18",
    freeze_grads: bool = False,
    outputs: int = 1,
    pretrained: bool = False,
    weights: Optional[str] = None,
):
    if (weights is not None) and weights:
        model = torch.load(weights, map_location="cpu")
    else:
        if model_name == "resnet18":
            pretrained_weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=pretrained_weights)
        elif model_name == "resnet50":
            pretrained_weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=pretrained_weights)
        else:
            raise Exception(
                f"Unsupported model_name, expected resnet18 or resnet50, got {model_name}"
            )

        if freeze_grads:
            for params in model.parameters():
                params.requires_grad_ = False

        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=outputs, bias=True
        )

    model.to(torch_config.device)

    return model
