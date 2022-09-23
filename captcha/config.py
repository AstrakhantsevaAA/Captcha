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


class NetConfig:
    symbols = string.ascii_letters + string.digits
    LEN_CAPTCHA = 5
    LEN_TOTAL = len(symbols) * LEN_CAPTCHA


system_config = SystemConfig()
torch_config = TorchConfig()
net_config = NetConfig()
