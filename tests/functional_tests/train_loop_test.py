import torch.optim as optim
from omegaconf import OmegaConf
from torch import nn

from captcha.config import net_config
from captcha.nets.define_net import define_net
from captcha.training.train import train_model, train_one_epoch
from captcha.training.train_utils import Phase


def test_loss_decreasing(test_dataloader):
    model = define_net(outputs=net_config.LEN_TOTAL, pretrained=True)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss1 = train_one_epoch(
        model, test_dataloader[Phase.train], optimizer, criterion, 1, None
    )

    loss2 = train_one_epoch(
        model, test_dataloader[Phase.train], optimizer, criterion, 2, None
    )

    assert loss1 > loss2


def test_deterministic(data_dir, test_csv):
    conf = OmegaConf.create(
        {
            "train": {
                "log_clearml": False,
                "epochs": 3,
                "task_name": "test",
                "model_save_path": "",
                "optimizer_name": "adam",
            },
            "dataloader": {
                "data_dir": data_dir,
                "train_path": test_csv,
                "eval_path": test_csv,
                "test_path": test_csv,
                "augmentations_intensity": 0,
                "test_size": 10,
                "batch_size": 2,
            },
            "net": {"continue_weights": "", "freeze_grads": False, "pretrained": False},
        }
    )
    loss1 = train_model(conf)
    loss2 = train_model(conf)

    assert loss1 == loss2
