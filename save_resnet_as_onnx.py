import torch
from torchvision import models, transforms
from PIL import Image

from settings import RESNET_IMAGE_SIZE, ONNX_MODEL_PATH, TEST_IMAGE_PATH

if __name__ == "__main__":
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(weights=True)
    model.eval()  # Set the model to evaluation mode

    # This image transform is just used to create an example image to base the dummy input on:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(RESNET_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess an image
    image: Image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    dummy_input = transform(image).unsqueeze(0)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
