import json
from pathlib import Path

import onnxruntime as ort
from PIL import Image
import numpy as np
from settings import RESNET_LABELS_JSON_PATH, ONNX_MODEL_PATH, TEST_IMAGE_PATH

SESSION = ort.InferenceSession(ONNX_MODEL_PATH)


def pre_process_image(img: Image) -> np.float32:
    input_size: tuple[int, int] = (224, 224)
    img = img.resize(input_size)
    img_array: np.float32 = np.array(img)
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    img_array = (img_array / 255.0 - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.array(img_array).astype(np.float32)
    return img_array


def get_inference_from_onnx(image_path: Path) -> str:
    image: Image = Image.open(image_path).convert("RGB")
    # Load Class Labels:
    with open(RESNET_LABELS_JSON_PATH, "r") as fs:
        imagenet_classes = json.load(fs)
    image_array = pre_process_image(image)
    input_name = SESSION.get_inputs()[0].name
    outputs = SESSION.run(None, {input_name: image_array})
    return imagenet_classes[outputs[0].argmax()]


def test_get_inference_from_onnx():
    assert "Retriever" in get_inference_from_onnx(TEST_IMAGE_PATH)
