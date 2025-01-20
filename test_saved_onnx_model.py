import json
from pathlib import Path


import onnxruntime as ort
from PIL import Image
import numpy as np
from save_resnet_as_onnx import onnx_file_path
from settings import RESNET_LABELS_JSON

# Start inference session:
session = ort.InferenceSession(onnx_file_path)

# Load Class Labels:
with open(RESNET_LABELS_JSON, "r") as fs:
    imagenet_classes = json.load(fs)

# Load image:
image_path: Path = Path(__file__).parent / "test-image-3.jpg"
image: Image = Image.open(image_path).convert("RGB")


def pre_process_image(image: Image) -> np.float32:
    input_size: tuple[int, int] = (224, 224)
    image = image.resize(input_size)
    image_array: np.float32 = np.array(image)
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    image_array = (image_array / 255.0 - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.array(image_array).astype(np.float32)
    return image_array


# Perform inference
image_array = pre_process_image(image)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_array})

# Get predicted class index
predicted_class = outputs[0].argmax()
print(f"Predicted class index: {predicted_class}")

# Print class name
class_name = imagenet_classes[predicted_class]
print(f"Predicted class index: {predicted_class}, class name: {class_name}")
