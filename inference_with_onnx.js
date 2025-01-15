import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';

async function preprocessImage(imageElement) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  canvas.width = 224;
  canvas.height = 224;

  // Resize and draw image to canvas
  ctx.drawImage(imageElement, 0, 0, 224, 224);

  const imageData = ctx.getImageData(0, 0, 224, 224).data;
  const floatImage = new Float32Array(224 * 224 * 3);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    floatImage[i * 3] = (imageData[i * 4] / 255.0 - mean[0]) / std[0];  // Red
    floatImage[i * 3 + 1] = (imageData[i * 4 + 1] / 255.0 - mean[1]) / std[1];  // Green
    floatImage[i * 3 + 2] = (imageData[i * 4 + 2] / 255.0 - mean[2]) / std[2];  // Blue
  }

  return {
    data: floatImage,
    dims: [1, 3, 224, 224]
  };
}

const InferenceSession = ort.InferenceSession;


async function loadModel(modelPath) {
    // Fetch the model (you can replace modelPath with the path to your model)
    const response = await fetch(modelPath);
    if (!response.ok) {
        throw new Error(`Failed to load model from ${modelPath}`);
    }
    const modelArrayBuffer = await response.arrayBuffer();

    // Create an inference session
    const session = await ort.InferenceSession.create(modelArrayBuffer);
    if (!session.ok){
      throw new Error("Failed to create session");
    }

    console.log("Model loaded successfully");

    // Now you can run inference with this session
    return session;
}



document.getElementById('uploadImage').addEventListener('change', async (event) => {
  // Debug to verify import of ort:
    const file = event.target.files[0];


    const imageElement = new Image();
    const session = await InferenceSession.create('./onnx_model/resnet.onnx')


  // imageElement.onload = async () => {
  //   try {
  //     const inputTensor = await preprocessImage(imageElement);
  //     const session = await loadModel('./onnx_model/resnet.onnx');
  //
  //     const results = await session.run({ input: inputTensor });
  //     const output = results.output.data;
  //     const predictedClassIndex = output.indexOf(Math.max(...output));
  //
  //     const response = await fetch('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json');
  //     const imagenetClasses = await response.json();
  //
  //     alert(`Predicted Class: ${imagenetClasses[predictedClassIndex]}`);
  //   } catch (err) {
  //     console.error('Error running model:', err);
  //   }
  // };

  imageElement.src = URL.createObjectURL(file);
});
