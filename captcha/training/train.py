from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from clearml import Logger, Task
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from captcha.config import net_config, system_config, torch_config
from captcha.nets.define_net import define_net
from captcha.training.train_utils import (
    Phase,
    create_dataloader,
    define_optimizer,
    fix_seeds,
)


def evaluation(
    model,
    eval_dataloader: DataLoader,
    criterion: Optional = None,
    epoch: int = -1,
    logger: Optional[Logger] = None,
    inference: bool = False,
    phase: Any = "val",
) -> pd.DataFrame:

    preds_collector = []
    model.eval()
    running_loss = 0.0
    iters = len(eval_dataloader)

    if inference:
        print("Gathering predictions...")
    else:
        print(f"Starting {phase} epoch {epoch}")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            logits = model.forward(batch["image"].to(torch_config.device))
            probs = nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().detach().numpy()

            batch_size = len(batch["image"])
            preds_batch = np.zeros((batch_size, net_config.LEN_CAPTCHA))
            if batch.get("label") is not None:
                labels_batch = np.zeros((batch_size, net_config.LEN_CAPTCHA))
            for i in range(net_config.LEN_CAPTCHA):
                start, end = (
                    i * net_config.LEN_SYMBOLS,
                    (i + 1) * net_config.LEN_SYMBOLS,
                )
                pred_max_idxs = np.argmax(probs[:, start:end], axis=1)
                if batch.get("label") is not None:
                    label_max_ids = np.argmax(batch["label"][:, start:end], axis=1)
                for j in range(batch_size):
                    preds_batch[j, i] = pred_max_idxs[j]
                    if batch.get("label") is not None:
                        labels_batch[j, i] = label_max_ids[j]

            preds_df = pd.DataFrame(
                preds_batch.tolist(),
                columns=[f"preds_{k}" for k in range(net_config.LEN_CAPTCHA)],
            )
            if batch.get("label") is not None:
                labels_df = pd.DataFrame(
                    labels_batch.tolist(),
                    columns=[f"labels_{k}" for k in range(net_config.LEN_CAPTCHA)],
                )
                preds_collector.append(pd.concat([preds_df, labels_df], axis=1))
            else:
                preds_collector.append(preds_df)

            if criterion is not None:
                loss = criterion(logits, batch["label"].to(torch_config.device))
                running_loss += loss.item()

    eval_preds_df = pd.concat(preds_collector, ignore_index=True)

    if not inference:
        accuracy = 0
        for l in range(net_config.LEN_CAPTCHA):
            accuracy += accuracy_score(
                eval_preds_df.iloc[:, l].values,
                eval_preds_df.iloc[:, net_config.LEN_CAPTCHA + l].values,
            )
        print(accuracy / net_config.LEN_CAPTCHA)

        if logger is not None:
            logger.report_scalar(
                f"Accuracy",
                phase,
                iteration=epoch,
                value=accuracy / net_config.LEN_CAPTCHA,
            )
            logger.report_scalar(
                f"Loss", phase, iteration=epoch, value=running_loss / iters
            )

    return eval_preds_df


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
        cfg.net.freeze_grads,
        outputs=net_config.LEN_TOTAL,
        pretrained=cfg.net.pretrained,
        weights=cfg.net.continue_weights,
    )

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = define_optimizer(cfg.train.optimizer_name, model)
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
