from collections import defaultdict
from enum import Enum
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from captcha.config import system_config
from captcha.data_utils.dataset import CaptchaDataset


class Phase(Enum):
    train = "train"
    val = "val"
    test = "test"


def create_dataloader(augmentations_intensity: float = 0, batch_size: int = 32):
    data_path = Path(
        "/home/alenaastrakhantseva/PycharmProjects/Captcha/data/raw/source6"
    )
    files = sorted(data_path.rglob("*.png"))

    df = pd.DataFrame({"filepath": files})
    df["label"] = df["filepath"].apply(lambda x: x.stem)

    dataloader = defaultdict()
    x_train, x_eval, y_train, y_eval = train_test_split(
        df["filepath"], df["label"], test_size=0.25, random_state=42, shuffle=True
    )

    shuffle = True
    for phase in Phase:
        if phase == Phase.val:
            augmentations_intensity, shuffle = 0.0, False
            x_train, y_train = x_eval, y_eval

        dataset = CaptchaDataset(
            system_config.data_dir, x_train, y_train, augmentations_intensity
        )
        dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
