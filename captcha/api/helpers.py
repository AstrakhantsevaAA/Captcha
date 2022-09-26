from typing import Optional

import numpy as np
import pandas as pd
import torch
from clearml import Logger
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from captcha.config import net_config, system_config, torch_config
from captcha.nets.define_net import define_net
from captcha.training.train_utils import create_dataloader


def evaluation(
    model,
    eval_dataloader: DataLoader,
    criterion: Optional = None,
    epoch: int = -1,
    logger: Optional[Logger] = None,
    inference: bool = False,
) -> pd.DataFrame:
    preds_collector = []
    model.eval()
    running_loss = 0.0
    iters = len(eval_dataloader)

    print(f"Starting validation epoch {epoch}")
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
                "eval",
                iteration=epoch,
                value=accuracy / net_config.LEN_CAPTCHA,
            )
            logger.report_scalar(
                f"Loss", "eval", iteration=epoch, value=running_loss / iters
            )

    return eval_preds_df


def inference(data: list):
    model = define_net(
        weights=str(system_config.model_dir / net_config.model_path),
    )
    loader = create_dataloader(batch_size=len(data), inference=True, files=data)
    preds = evaluation(model, loader, inference=True)

    predictions = preds.iloc[:, : net_config.LEN_CAPTCHA].values.tolist()
    labels = preds.iloc[:, net_config.LEN_CAPTCHA :].values.tolist()
    decode_prediction = [[net_config.symbols[int(idx)] for idx in p] for p in predictions]
    decode_labels = [[net_config.symbols[int(idx)] for idx in l] for l in labels]
    external_data = {
        "id": 0,
        "predictions": predictions,
        "labels": labels,
        "decode_prediction": decode_prediction,
        "decode_labels": decode_labels,
    }
    return external_data
