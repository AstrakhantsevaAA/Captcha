from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (
    Affine,
    Blur,
    CoarseDropout,
    Compose,
    Downscale,
    GridDistortion,
    PadIfNeeded,
    Perspective,
    ShiftScaleRotate,
)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from captcha.config import net_config
from captcha.data_utils.preprocessing import image_preprocess


def one_hot(letter: str) -> list:
    oh_label = np.zeros(len(net_config.symbols))
    idx = net_config.symbols.find(letter)
    oh_label[idx] = 1
    return list(oh_label)


class CaptchaDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image tensors, and label.
    """

    def __init__(
        self,
        data_dir: Path,
        x_df: pd.DataFrame,
        augmentations_intensity: float = 0.0,
        test_size: int = 0,
    ):
        self.data_dir = data_dir
        self.data = x_df if test_size <= 0 else x_df.iloc[:test_size]
        if "label" in self.data.columns:
            self.labels = self.data["label"].astype(str)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        if augmentations_intensity > 0:
            self.augmentation = Compose(
                [
                    CoarseDropout(
                        max_height=3,
                        max_width=3,
                        p=0.99,
                        min_holes=20,
                        max_holes=40,
                        fill_value=255,
                    ),
                    CoarseDropout(
                        max_height=3,
                        max_width=3,
                        p=0.99,
                        min_holes=20,
                        max_holes=100,
                        fill_value=0,
                    ),
                    Blur(blur_limit=3, p=0.1),
                    ShiftScaleRotate(p=1.0, rotate_limit=20),
                    Affine(),
                    GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
                    Downscale(
                        scale_min=0.6,
                        scale_max=0.9,
                        p=0.7,
                    ),
                    Perspective(p=0.9),
                    PadIfNeeded(
                        min_height=180,
                        min_width=200,
                        always_apply=True,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                ],
                p=augmentations_intensity,
            )
        else:
            self.augmentation = Compose(
                [
                    PadIfNeeded(
                        min_height=180,
                        min_width=200,
                        always_apply=True,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                ]
            )

    def __getitem__(self, index):
        filepath = str(self.data["filepath"].iloc[index])
        image = cv2.imread(filepath)

        if image is None:
            raise Exception(
                f"image is None, got filepath: {filepath} \n data: {self.data}"
            )

        image = image_preprocess(image)

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        image = self.transform(Image.fromarray(image))

        if self.labels is None:
            sample = {"image": image}
        else:
            oh_label = []
            label = self.labels.iloc[index]
            for l in label:
                oh_label += one_hot(l)
            label_encode = torch.tensor(oh_label, dtype=torch.float)
            assert (
                len(label_encode) == net_config.LEN_TOTAL
            ), f"{len(label_encode)}, {label}"
            sample = {
                "image": image,
                "label": label_encode,
                "label_decode": label,
                "filepath": filepath,
            }

        return sample

    def __len__(self):
        return len(self.data)
