from fastapi import FastAPI
from fastapi.responses import FileResponse

from settings import ONNX_MODEL_PATH, RESNET_LABELS_JSON_PATH

app = FastAPI()


@app.get("/model", response_class=FileResponse)
async def get_model():
    return ONNX_MODEL_PATH


@app.get("/labels", response_class=FileResponse)
async def get_labels():
    return RESNET_LABELS_JSON_PATH
