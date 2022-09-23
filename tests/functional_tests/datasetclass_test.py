from pathlib import Path

import pandas as pd

from captcha.config import net_config
from captcha.data_utils.dataset import CaptchaDataset
from captcha.training.train_utils import Phase, create_dataloader


def test_one_sample():
    data_path = Path(
        "/home/alenaastrakhantseva/PycharmProjects/Captcha/data/raw/source6"
    )
    files = sorted(data_path.rglob("*.png"))

    df = pd.DataFrame({"filepath": files})
    df["label"] = df["filepath"].apply(lambda x: x.stem)

    dataset = CaptchaDataset(
        Path("/"),
        df["filepath"],
        df["label"],
    )
    sample = dataset[1]

    print(sample["image"].shape)
    print(sample["label"])

    assert len(sample["image"].shape) == 3
    assert sample["image"].shape[0] == 3
    assert sample["label"].shape[0] == net_config.LEN_TOTAL


def test_dataloader():
    batch_size = 32
    dataloader = create_dataloader(augmentations_intensity=0.5)
    for batch in dataloader[Phase.train]:
        image = batch["image"]
        label = batch["label"]

        print(label[0])

        assert len(image) == batch_size
        break
