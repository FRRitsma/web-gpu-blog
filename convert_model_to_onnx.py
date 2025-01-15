from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
import requests

onnx_file_path: Path = Path(__file__).parent / "onnx_model" / "resnet.onnx"

# Load a pre-trained ResNet-18 model
model = models.resnet18(weights=True)
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
# image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"  # Replace with your image URL
image_url = "https://yucatantoday.com/hubfs/Imported_Blog_Media/Pavo-real-4.jpg"  # Replace with your image URL"  # Replace with your image URL
image = Image.open(requests.get(image_url, stream=True).raw)
dummy_input = transform(image).unsqueeze(0)

torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,  # Store the trained parameter weights
    opset_version=11,    # Specify ONNX opset version
    do_constant_folding=True,  # Optimize the model by folding constants
    input_names=['input'],  # Name of the input tensor
    output_names=['output'],  # Name of the output tensor
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Support dynamic batching
)
