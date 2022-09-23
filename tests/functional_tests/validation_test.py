import torch.optim as optim
from torch import nn

from captcha.config import net_config
from captcha.nets.define_net import define_net
from captcha.training.train import evaluation, train_one_epoch
from captcha.training.train_utils import Phase, create_dataloader, fix_seeds


def test_validation():
    fix_seeds()
    dataloader = create_dataloader(test_size=10, batch_size=2)
    model = define_net(outputs=net_config.LEN_TOTAL, pretrained=True)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    _ = train_one_epoch(model, dataloader[Phase.train], optimizer, criterion, 1, None)
    evaluation(model, dataloader[Phase.val], criterion, epoch=1)
