import json
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
import requests

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
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"  # Replace with your image URL
# image_url = "https://yucatantoday.com/hubfs/Imported_Blog_Media/Pavo-real-4.jpg"  # Replace with your image URL"  # Replace with your image URL
image = Image.open(requests.get(image_url, stream=True).raw)
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Get predicted class index
_, predicted_class = output.max(1)
print(f"Predicted class index: {predicted_class.item()}")


# Load ImageNet class labels
imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
).json()

with open(Path(__file__).parent / "resnet_labels.json", "w") as fs:
    json.dump(imagenet_classes, fs)


# Print class name
class_name = imagenet_classes[predicted_class.item()]
print(f"Predicted class index: {predicted_class.item()}, class name: {class_name}")