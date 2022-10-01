import pandas as pd

from captcha.api.helpers import inference


def test_inference(files):
    response = inference(files)
    print(response)
    pd.DataFrame(response).to_csv("outputs/test_output.csv")
