import os
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from captcha.config import system_config
from captcha.data_utils.dataset import CaptchaDataset


class Phase(Enum):
    train = "train"
    val = "val"
    test = "test"


def fix_seeds(random_state: int = 42):
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(
    data_dir: Path = system_config.data_dir,
    csv_path: Optional[Union[Path, str, list, dict]] = None,
    augmentations_intensity: float = 0,
    batch_size: int = 32,
    test_size: int = 0,
    inference: bool = False,
    files: Optional = None,
):
    dataloader = defaultdict()

    if inference:
        if files is None:
            raise Exception("files is None, for inference files are necessary")
        df = pd.DataFrame({"filepath": files})
        df["label"] = df["filepath"].apply(lambda x: Path(x).stem)
        dataset = CaptchaDataset(data_dir, df)

        return DataLoader(dataset, batch_size=batch_size)

    if csv_path is None or not csv_path:
        raise Exception(
            "csv files with train and eval data are None, for training those files are necessary"
        )

    data = defaultdict()
    if isinstance(csv_path, str) or isinstance(csv_path, Path):
        for phase in Phase:
            data[phase] = pd.read_csv(data_dir / csv_path, dtype={"label": str})
    elif isinstance(csv_path, list):
        for path, phase in zip(csv_path, Phase):
            data[phase] = pd.read_csv(data_dir / path, dtype={"label": str})
    elif isinstance(csv_path, dict):
        for phase in Phase:
            if csv_path.get(phase.value) is not None:
                data[phase] = pd.read_csv(
                    data_dir / csv_path.get(phase.value), dtype={"label": str}
                )

    shuffle = True
    for phase in Phase:
        if phase == Phase.val:
            augmentations_intensity, shuffle = augmentations_intensity, False

        if phase == Phase.test:
            augmentations_intensity, shuffle = 0.0, False

        if data.get(phase) is not None:
            dataset = CaptchaDataset(
                data_dir,
                data[phase],
                augmentations_intensity,
                test_size=test_size,
            )
            dataloader[phase] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )

    return dataloader


def define_optimizer(optimizer_name: str, model):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.003)
    elif optimizer_name == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=0.003)
    else:
        raise Exception(
            f"Wrong optimizer name! Expected 'sgd' or 'adam', got {optimizer_name}"
        )

    return optimizer
