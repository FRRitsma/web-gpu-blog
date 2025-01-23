from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from settings import ONNX_MODEL_PATH, RESNET_LABELS_JSON_PATH

app = FastAPI()





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use a specific origin in production, e.g., ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/model", response_class=FileResponse)
async def get_model():
    return ONNX_MODEL_PATH


@app.get("/labels", response_class=FileResponse)
async def get_labels():
    return RESNET_LABELS_JSON_PATH
