from pathlib import Path

from captcha.api.helpers import inference


def test_inference():
    files = [
        Path(
            "/home/alenaastrakhantseva/PycharmProjects/Captcha/data/raw/source6/Large_Captcha_Dataset/0a954.png"
        )
    ]
    response = inference(files)
    print(response)
