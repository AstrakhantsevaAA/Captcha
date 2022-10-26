import copy
from typing import Optional

import timm
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
        elif model_name == "resnest-50":
            torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
            model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
        elif model_name == "rexnet-100":
            model = timm.create_model('rexnet_100', pretrained=pretrained)
        elif model_name == "convnext":
            model = timm.create_model('convnext_base')
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth", check_hash=True)
            model_weights = copy.deepcopy(checkpoint["model"])
            model.load_state_dict(model_weights)

        else:
            raise Exception(
                f"Unsupported model_name, expected resnet18, resnet50, resnest-50, rexnet-100, got {model_name}"
            )

        if freeze_grads:
            for params in model.parameters():
                params.requires_grad_ = False

        if model_name == "rexnet-100":
            model.head.fc = nn.Linear(
                in_features=model.head.fc.in_features, out_features=outputs, bias=True
            )
        else:
            model.fc = nn.Linear(
                in_features=model.fc.in_features, out_features=outputs, bias=True
            )

    model.to(torch_config.device)

    return model
