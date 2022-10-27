import os
import string
from pathlib import Path

import torch


class SystemConfig:
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / "models"
    data_dir = root_dir / "data"
    raw_data_dir = data_dir / "raw"


class TorchConfig:
    if os.getenv("FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


class NetConfig:
    symbols = string.digits
    LEN_CAPTCHA = 6
    LEN_SYMBOLS = len(symbols)
    LEN_TOTAL = LEN_SYMBOLS * LEN_CAPTCHA
    model_path = "release_v0.3.0/model.pth"


system_config = SystemConfig()
torch_config = TorchConfig()
net_config = NetConfig()
