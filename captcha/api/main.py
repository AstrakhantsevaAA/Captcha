from enum import Enum
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

from captcha.api.helpers import inference


class APIResponse(BaseModel):
    id: int
    endpoint_name: str = "classification"
    predictions: Optional[List[List[float]]]
    labels: Optional[List[List[float]]]
    other: Optional[str]
    message: str = "Successful"


class Data(BaseModel):
    image_paths: List[str]


class EndpointName(str, Enum):
    classification = "classification"


app = FastAPI()


@app.post("/endpoints/{endpoint_name}")
async def get_model(endpoint_name: EndpointName, data: Data):
    external_data = {}
    if endpoint_name == EndpointName.classification:
        external_data = inference(data.image_paths)
    try:
        response = APIResponse(**external_data)
    except ValidationError as e:
        response = APIResponse(id=0, message=str(e))

    return response
