import os
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
    augmentations_intensity: float = 0,
    batch_size: int = 32,
    test_size: int = 0,
    inference: bool = False,
    files: Optional = None,
):
    dataloader = defaultdict()

    if files is None:
        data_path = Path(
            "/home/alenaastrakhantseva/PycharmProjects/Captcha/data/raw/source6"
        )
        files = sorted(data_path.rglob("*.png"))

    df = pd.DataFrame({"filepath": files})
    df["label"] = df["filepath"].apply(lambda x: x.stem)

    if inference:
        dataset = CaptchaDataset(system_config.data_dir, df["filepath"], df["label"])
        return DataLoader(dataset, batch_size=batch_size)

    x_train, x_eval, y_train, y_eval = train_test_split(
        df["filepath"], df["label"], test_size=0.25, random_state=42, shuffle=True
    )

    shuffle = True
    for phase in Phase:
        if phase == Phase.val:
            augmentations_intensity, shuffle = 0.0, False
            x_train, y_train = x_eval, y_eval

        dataset = CaptchaDataset(
            system_config.data_dir,
            x_train,
            y_train,
            augmentations_intensity,
            test_size=test_size,
        )
        dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
