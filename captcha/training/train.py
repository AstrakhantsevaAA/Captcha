from pathlib import Path
from typing import Any, Optional

import hydra
import torch
from clearml import Logger, Task
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from captcha.api.helpers import evaluation
from captcha.config import net_config, system_config, torch_config
from captcha.nets.define_net import define_net
from captcha.training.train_utils import (
    Phase,
    create_dataloader,
    define_optimizer,
    fix_seeds,
)


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer: Any,
    criterion: Any,
    epoch: int,
    logger: Optional[Logger],
):
    model.train()
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
        logger.report_scalar("Loss", "train", iteration=epoch, value=loss)
        logger.report_scalar(
            "LR", "train", iteration=epoch, value=optimizer.param_groups[0]["lr"]
        )

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
        data_dir=Path(cfg.dataloader.data_dir),
        csv_path=[
            cfg.dataloader.train_path,
            cfg.dataloader.eval_path,
            cfg.dataloader.test_path,
        ],
        augmentations_intensity=cfg.dataloader.augmentations_intensity,
        batch_size=cfg.dataloader.batch_size,
        test_size=cfg.dataloader.test_size,
    )

    model = define_net(
        model_name=cfg.net.model_name,
        freeze_grads=cfg.net.freeze_grads,
        outputs=net_config.LEN_TOTAL,
        pretrained=cfg.net.pretrained,
        weights=cfg.net.continue_weights,
    )

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = define_optimizer(cfg.train.optimizer_name, model)
    if cfg.scheduler.scheduler:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.t0,
            T_mult=cfg.scheduler.t_mult,
            eta_min=0.000001,
        )
    else:
        scheduler = None
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
        if dataloader.get(Phase.test) is not None:
            _ = evaluation(
                model,
                dataloader[Phase.test],
                criterion,
                epoch,
                logger,
                phase=Phase.test.value,
            )
        if scheduler:
            scheduler.step()

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
