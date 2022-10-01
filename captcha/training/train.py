from typing import Any, Optional

import hydra
import torch
import torch.optim as optim
from clearml import Logger, Task
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from captcha.api.helpers import evaluation
from captcha.config import net_config, system_config, torch_config
from captcha.nets.define_net import define_net
from captcha.training.train_utils import Phase, create_dataloader, fix_seeds


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer: Any,
    criterion: Any,
    epoch: int,
    logger: Optional[Logger],
):
    running_loss = 0
    iters = len(dataloader)
    print(f"Starting training epoch {epoch}")
    for batch_n, batch in tqdm(enumerate(dataloader), total=iters):
        optimizer.zero_grad()
        outputs = model(batch["image"].to(torch_config.device))
        loss = criterion(outputs, batch["label"].to(torch_config.device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if logger is not None:
            logger.report_scalar(
                f"Running_loss",
                "train",
                iteration=(epoch + 1) * batch_n,
                value=running_loss / (batch_n + 1),
            )
    loss = running_loss / iters
    if logger is not None:
        logger.report_scalar(f"Loss", "train", iteration=epoch, value=loss)

    return loss


@hydra.main(version_base=None, config_path=".", config_name="config")
def train_model(cfg: DictConfig):
    task = (
        Task.init(project_name="captcha", task_name=cfg.train.task_name)
        if cfg.train.log_clearml
        else None
    )
    logger = None if task is None else task.get_logger()
    fix_seeds()

    dataloader = create_dataloader(
        cfg.train.train_path,
        cfg.train.eval_path,
        cfg.train.augmentations_intensity,
        cfg.train.batch_size,
        cfg.train.test_size,
    )

    model = define_net(
        cfg.net.freeze_grads,
        outputs=net_config.LEN_TOTAL,
        pretrained=True,
    )

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss = 0.0

    for epoch in range(cfg.train.epochs):
        loss = train_one_epoch(
            model, dataloader[Phase.train], optimizer, criterion, epoch, logger
        )
        _ = evaluation(
            model,
            dataloader[Phase.val],
            criterion,
            epoch,
            logger,
            phase=Phase.val.value,
        )
        _ = evaluation(
            model,
            dataloader[Phase.test],
            criterion,
            epoch,
            logger,
            phase=Phase.test.value,
        )
        if cfg.train.model_save_path:
            model_save_path = system_config.model_dir / cfg.train.model_save_path
            model_save_path.mkdir(exist_ok=True, parents=True)
            print(f"Saving model to {model_save_path} as model.pth")
            torch.save(
                model, system_config.model_dir / cfg.train.model_save_path / "model.pth"
            )

    if logger is not None:
        logger.flush()

    return loss


if __name__ == "__main__":
    train_model()
