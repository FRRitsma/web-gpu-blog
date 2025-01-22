from pathlib import Path

DIRECTORY_ROOT_PATH = Path(__file__).parent
RESNET_LABELS_JSON_PATH = DIRECTORY_ROOT_PATH / "resnet_labels.json"
ONNX_MODEL_PATH = DIRECTORY_ROOT_PATH / "onnx_model" / "resnet.onnx"
