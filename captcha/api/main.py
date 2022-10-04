from enum import Enum
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, ValidationError

from captcha.api.helpers import inference


class APIResponse(BaseModel):
    endpoint_name: str = "classification"
    predictions: Optional[List[List[float]]]
    labels: Optional[List[List[float]]]
    decode_prediction: Optional[List[List[str]]]
    decode_labels: Optional[List[List[str]]]
    other: Optional[str]
    message: str = "Successful"


class Data(BaseModel):
    images: list[UploadFile]


class EndpointName(str, Enum):
    classification = "classification"


app = FastAPI()


@app.get("/health_check")
def health_check():
    return APIResponse(endpoint_name="health_check", message="OK")


@app.post("/{endpoint_name}")
async def get_model(endpoint_name: EndpointName, data: UploadFile):
    data = check_data([data])
    external_data = {}
    if endpoint_name == EndpointName.classification:
        external_data = inference(data)
    try:
        response = APIResponse(**external_data)
    except ValidationError as e:
        response = APIResponse(message=str(e))

    return response


def check_data(files):
    new_paths = []
    for file in files:
        try:
            contents = file.file.read()
            new_filepath = Path("/tmp/captcha")
            new_filepath.mkdir(parents=True, exist_ok=True)
            print(new_filepath / file.filename)
            with open(new_filepath / file.filename, "wb") as f:
                f.write(contents)
            new_paths.append(new_filepath / file.filename)
        except Exception as e:
            return {"message": f"There was an error uploading the file(s).\nError: {e}"}
        finally:
            file.file.close()

    return new_paths
