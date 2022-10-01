import torch.optim as optim
from torch import nn

from captcha.config import net_config
from captcha.nets.define_net import define_net
from captcha.training.train import evaluation, train_one_epoch
from captcha.training.train_utils import Phase


def test_validation(test_dataloader):
    model = define_net(outputs=net_config.LEN_TOTAL, pretrained=True)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    _ = train_one_epoch(
        model, test_dataloader[Phase.train], optimizer, criterion, 1, None
    )
    preds = evaluation(model, test_dataloader[Phase.val], criterion, epoch=1)
    print(preds)
