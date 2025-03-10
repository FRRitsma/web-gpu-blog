Goal: 

Demonstrate how to save a model as an ONNX model and transfer it to a completely different framework, such as javascript.

This can then be used to perform client side inference.


save_resnet_as_onnx.py demonstrates how to convert a pytorch model to the onnx format
inference_on_onnx.py demonstrates how to use a saved onnx model to perform inference with


Running the app:

(After model has been downloaded and converted to .onnx file)

1. Activate virtual environment
2. Run `uvicorn main:app` to serve fastapi
3. Run `test.html`, this should open your default browser
4. Click "Choose File" and open an image file. Images are included in test_images
5. Click "Run Inference" and await the results