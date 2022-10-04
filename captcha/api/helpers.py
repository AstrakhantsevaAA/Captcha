from captcha.config import net_config, system_config
from captcha.nets.define_net import define_net
from captcha.training.train import evaluation
from captcha.training.train_utils import create_dataloader


def inference(data: list):
    model = define_net(
        weights=str(system_config.model_dir / net_config.model_path),
    )
    loader = create_dataloader(
        batch_size=len(data), inference=True, files=data, augmentations_intensity=0.0
    )
    preds = evaluation(model, loader, inference=True)

    predictions = preds.iloc[:, : net_config.LEN_CAPTCHA].values.tolist()
    labels = preds.iloc[:, net_config.LEN_CAPTCHA :].values.tolist()
    decode_prediction = [
        [net_config.symbols[int(idx)] for idx in p] for p in predictions
    ]
    decode_labels = [[net_config.symbols[int(idx)] for idx in l] for l in labels]
    external_data = {
        "predictions": predictions,
        "labels": labels,
        "decode_prediction": decode_prediction,
        "decode_labels": decode_labels,
    }
    return external_data
