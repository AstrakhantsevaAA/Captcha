from pathlib import Path

import pandas as pd

from captcha.config import net_config
from captcha.data_utils.dataset import CaptchaDataset
from captcha.training.train_utils import Phase, create_dataloader


def test_datasetclass(files):
    df = pd.DataFrame({"filepath": files})
    df["label"] = df["filepath"].apply(lambda x: x.stem)

    dataset = CaptchaDataset(
        Path("/"),
        df,
    )
    sample = dataset[0]

    print(sample["image"].shape)
    print(sample["label"])
    print(sample["label_decode"])

    assert len(sample["image"].shape) == 3
    assert sample["image"].shape[0] == 1
    assert sample["label"].shape[0] == net_config.LEN_TOTAL


def test_dataloader(test_csv, data_dir):
    batch_size = 2
    dataloader = create_dataloader(
        data_dir=data_dir,
        csv_path=test_csv,
        augmentations_intensity=0.5,
        test_size=batch_size,
    )
    for batch in dataloader[Phase.train]:
        image = batch["image"]
        label = batch["label"]

        print(batch["label_decode"])
        print(batch["filepath"])
        print(label[0])

        assert len(image) == batch_size
        assert batch["label_decode"][0] == Path(batch["filepath"][0]).stem


def test_inference_dataloader(files):
    batch_size = 2
    dataloader = create_dataloader(
        files=files, augmentations_intensity=0.5, inference=True, batch_size=batch_size
    )

    for batch in dataloader:
        image = batch["image"]
        label = batch["label"]

        print(batch["label_decode"])
        print(batch["filepath"])
        print(label[0])
        print(image.shape)

        assert len(image) == batch_size
        assert batch["label_decode"][0] == Path(batch["filepath"][0]).stem
