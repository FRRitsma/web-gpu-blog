from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse
from fastapi import FastAPI
from settings import ONNX_MODEL_PATH, RESNET_LABELS_JSON_PATH, DIRECTORY_ROOT_PATH
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Define paths
template_name: str = "static_files"
template_dir: Path = DIRECTORY_ROOT_PATH / template_name

app.mount("/static", StaticFiles(directory=template_dir), name="static")

templates = Jinja2Templates(directory=str(template_dir))


@app.get("/", response_class=HTMLResponse)
async def read_items(request: Request):
    return templates.TemplateResponse(
        "minimal_interface.html", {"request": request, "title": "Home Page"}
    )


@app.get("/model", response_class=FileResponse)
async def get_model():
    return ONNX_MODEL_PATH


@app.get("/labels", response_class=FileResponse)
async def get_labels():
    return RESNET_LABELS_JSON_PATH
