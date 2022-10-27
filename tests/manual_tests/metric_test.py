import pandas as pd
import requests
import typer
from clearml import Task
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from captcha.config import net_config


def check_equality(pred: list, label: list, save_path: str):
    if pred != label:
        pd.DataFrame(
            {"pred": [", ".join(map(str, pred))], "label": [", ".join(map(str, label))]}
        ).to_csv(f"{save_path}/unequal.csv", index=False, header=False, mode="a")


def get_response(files: list, url: str):
    files_opened = []
    for file in files:
        files_opened.append(("data", open(file, "rb")))

    response = requests.post(
        url=f"{url}/predict",
        files=files_opened,
    )
    return response


def metric_test(
    csv_path: str,
    url: str,
    batch_size: int = 15,
    task_name: str = "metric_test",
    log_clearml: bool = True,
):
    task = (
        Task.init(
            project_name="captcha_metrics",
            task_name=f"{task_name}_{net_config.model_path}",
        )
        if log_clearml
        else None
    )
    logger = None if task is None else task.get_logger()

    outputs_path = f"outputs/{datetime.now()}"
    Path(outputs_path).mkdir(exist_ok=True, parents=True)

    files = list(pd.read_csv(csv_path)["filepath"].values)
    batches = [
        files[i * batch_size : (i + 1) * batch_size]
        for i in range(int(len(files) / batch_size))
    ]
    preds = []
    labels = []
    for batch in tqdm(batches, desc=f"batch prediction... (batch_size={batch_size})"):
        predictons = get_response(batch, url).json()
        for item in range(len(predictons["predictions"])):
            check_equality(predictons["predictions"][item], predictons["labels"][item], outputs_path)
            preds.extend(predictons["predictions"][item])
            labels.extend(predictons["labels"][item])

    accuracy = accuracy_score(labels, preds)
    print(f"data: {csv_path}\nmodel: {net_config.model_path}\naccuracy: {accuracy}")

    if logger is not None:
        logger.report_scalar(f"Accuracy", net_config.model_path, iteration=0, value=accuracy)
        logger.flush()


if __name__ == "__main__":
    typer.run(metric_test)
