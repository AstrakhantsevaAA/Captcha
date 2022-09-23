from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from albumentations import (
    Blur,
    CoarseDropout,
    Compose,
    Downscale,
    HorizontalFlip,
    OneOf,
    RandomRotate90,
    Rotate,
    ShiftScaleRotate,
    VerticalFlip,
)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from captcha.config import net_config


def one_hot(letter: str) -> list:
    oh_label = np.zeros(len(net_config.symbols))
    idx = net_config.symbols.find(letter)
    oh_label[idx] = 1
    return list(oh_label)


class CaptchaDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(
        self,
        data_dir: Path,
        x_df: pd.DataFrame,
        y_df: Optional[pd.DataFrame] = None,
        augmentations_intensity: float = 0.0,
    ):
        self.data_dir = data_dir
        self.data = x_df
        self.labels = y_df
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        if augmentations_intensity > 0:
            self.augmentation = Compose(
                [
                    OneOf(
                        [HorizontalFlip(), VerticalFlip()],
                        p=0.1,
                    ),
                    OneOf(
                        [Rotate(p=1.0, limit=30), RandomRotate90(p=1.0)],
                        p=0.1,
                    ),
                    ShiftScaleRotate(p=1.0, rotate_limit=30),
                    Blur(blur_limit=3, p=0.1),
                    CoarseDropout(max_height=7, max_width=7, p=0.1),
                    Downscale(
                        scale_min=0.6,
                        scale_max=0.9,
                        p=0.7,
                    ),
                ],
                p=augmentations_intensity,
            )
        else:
            self.augmentation = None

    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index]).convert("RGB")
        if self.augmentation is not None:
            image = self.augmentation(image=np.array(image))["image"]
            image = self.transform(Image.fromarray(image))
        else:
            image = self.transform(image)
        image_id = self.data.index[index]

        if self.labels is None:
            sample = {"image_id": image_id, "image": image}
        else:
            oh_label = []
            label = self.labels.iloc[index]
            for l in label:
                oh_label += one_hot(l)
            label = torch.tensor(oh_label, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}

        return sample

    def __len__(self):
        return len(self.data)
