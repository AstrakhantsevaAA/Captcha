from pathlib import Path

import pytest

from captcha.training.train_utils import create_dataloader, fix_seeds


@pytest.fixture
def data_dir():
    tests_root_dir = Path(__file__).parent
    return tests_root_dir / "test_data"


@pytest.fixture
def files(data_dir):
    data = []
    extensions = ["jpeg", "png"]
    for ext in extensions:
        data.extend(sorted(data_dir.rglob(f"*.{ext}")))
    return data


@pytest.fixture
def test_csv(data_dir):
    return data_dir / "set.csv"


@pytest.fixture
def test_dataloader(data_dir, test_csv):
    fix_seeds()
    dataloader = create_dataloader(data_dir=data_dir, csv_path=test_csv)
    return dataloader
