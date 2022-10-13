from pathlib import Path

import pytest

from captcha.training.train_utils import create_dataloader, fix_seeds


@pytest.fixture
def data_dir() -> Path:
    tests_root_dir = Path(__file__).parent
    return tests_root_dir / "test_data"


@pytest.fixture
def files(data_dir: Path) -> list:
    data = []
    extensions = ["jpeg", "png"]
    for ext in extensions:
        data.extend(sorted(data_dir.rglob(f"*.{ext}")))
    return data


@pytest.fixture
def test_csv(data_dir: Path) -> Path:
    return data_dir / "set.csv"


@pytest.fixture
def test_dataloader(data_dir: Path, test_csv: Path):
    fix_seeds()
    dataloader = create_dataloader(data_dir=data_dir, csv_path=test_csv)
    return dataloader


@pytest.fixture
def url():
    return "http://127.0.0.1:8000"
