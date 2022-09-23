from pathlib import Path

import pandas as pd

from captcha.data_utils.dataset import CaptchaDataset, LEN_CAPTCHA, symbols


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
    assert sample["label"].shape[0] == LEN_CAPTCHA * len(symbols)
