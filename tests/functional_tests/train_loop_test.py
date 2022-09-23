import torch.optim as optim
from torch import nn
from omegaconf import OmegaConf

from captcha.config import net_config
from captcha.nets.define_net import define_net
from captcha.training.train import train_one_epoch, train_model
from captcha.training.train_utils import Phase, create_dataloader, fix_seeds


def test_loss_decreasing():
    fix_seeds()
    dataloader = create_dataloader(test_size=10, batch_size=2)
    model = define_net(outputs=net_config.LEN_TOTAL, pretrained=True)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss1 = train_one_epoch(
        model, dataloader[Phase.train], optimizer, criterion, 1, None
    )

    loss2 = train_one_epoch(
        model, dataloader[Phase.train], optimizer, criterion, 2, None
    )

    assert loss1 > loss2


def test_deterministic():
    conf = OmegaConf.create({
        "train": {
            "log_clearml": False,
            "epochs": 3,
            "task_name": "baseline",
            "augmentations_intensity": 0,
            "model_save_path": "",
            "test_size": 10,
            "batch_size": 2,
        },
        "net": {
            "freeze_grads": False,
        }
    })
    loss1 = train_model(conf)
    loss2 = train_model(conf)

    assert loss1 == loss2
