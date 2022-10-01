import os
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
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
    train_path: Optional[str] = None,
    eval_path: Optional[str] = None,
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
        dataset = CaptchaDataset(system_config.data_dir, df)

        return DataLoader(dataset, batch_size=batch_size)

    if train_path is None or eval_path is None:
        raise Exception(
            "csv files with train and eval data are None, for training those files are necessary"
        )

    data = pd.read_csv(
        system_config.data_dir / train_path, index_col=0, dtype={"label": str}
    )
    data_eval = pd.read_csv(
        system_config.data_dir / eval_path, index_col=0, dtype={"label": str}
    )
    data_test = pd.read_csv(
        system_config.data_dir / "processed/test_set.csv", dtype={"label": str}
    )

    shuffle = True
    for phase in Phase:
        if phase == Phase.val:
            augmentations_intensity, shuffle = 0.9, False
            data = data_eval

        if phase == Phase.test:
            augmentations_intensity, shuffle = 0.0, False
            data = data_test

        dataset = CaptchaDataset(
            system_config.data_dir,
            data,
            augmentations_intensity,
            test_size=test_size,
        )
        dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
