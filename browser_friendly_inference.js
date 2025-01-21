const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

// Constants
const modelUrl = './onnx_model/resnet.onnx';
const labelsUrl = './resnet_labels.json';
const targetSize = 224;
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

let session;

// Load ONNX model
async function loadModel() {
  try {
    session = await InferenceSession.create(modelUrl);
    console.log('ONNX model loaded.');
    document.getElementById('runInference').disabled = false; // Enable button
  } catch (error) {
    console.error('Failed to load ONNX model:', error);
  }
}

// Load labels
async function loadLabels() {
  const response = await fetch(labelsUrl);
  return await response.json();
}

// Resize and preprocess the image
function preprocessImage(imageData) {
    // Convert image data to Float32 and normalize
    const floatImage = Float32Array.from(imageData, (pixel) => pixel / 255.0);
    console.assert(floatImage.length === targetSize*targetSize*3, "floatImage leads to an invalid buffer size");

    // Normalize channels (RGB)
    const normalizedImage = new Float32Array(floatImage.length);
    for (let i = 0; i < floatImage.length; i++) {
        const channel = i % 3; // R, G, B channels
        normalizedImage[i] = (floatImage[i] - mean[channel]) / std[channel];
    }

    // Rearrange dimensions from [H, W, C] to [C, H, W]
    const channels = 3;
    const height = targetSize;
    const width = targetSize;
    const transposedImage = new Float32Array(normalizedImage.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let c = 0; c < channels; c++) {
                transposedImage[c * height * width + y * width + x] =
                    normalizedImage[(y * width + x) * channels + c];
            }
        }
    }
    return new ort.Tensor('float32', transposedImage, [1, 3, height, width]);
}

// Resize image using canvas
async function resizeImage(imageFile) {
  const img = new Image();
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  // Validate that canvas context is available
  if (!ctx) {
    throw new Error("Failed to get 2D context from canvas.");
  }

  // Load the image
  const imageLoaded = new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("Failed to load the image."));
  });

  img.src = URL.createObjectURL(imageFile);
  await imageLoaded;

  // Set canvas dimensions and draw the image
  canvas.width = targetSize;
  canvas.height = targetSize;
  ctx.drawImage(img, 0, 0, targetSize, targetSize);

  // Ensure all parameters are integers before calling getImageData
  const x = 0;
  const y = 0;
  const width = targetSize;
  const height = targetSize;

  const imageData = ctx.getImageData(x, y, width, height).data;
  // Convert RGBA to RGB
  const rgbData = [];
  for (let i = 0; i < imageData.length; i += 4) {
    rgbData.push(imageData[i]);     // Red
    rgbData.push(imageData[i + 1]); // Green
    rgbData.push(imageData[i + 2]); // Blue
  }
  // Log the length of the RGB data array
  console.assert(rgbData.length === targetSize*targetSize*3, "Extracted image leads to an invalid buffer size");
  return rgbData;
}


// Run inference and display results
async function runInference(imageFile) {
  try {
    const imageData = await resizeImage(imageFile);
    const inputTensor = preprocessImage(imageData);
    console.log('Tensor shape:', inputTensor.dims); // Should log [1, 3, 224, 224]
    const inputName = session.inputNames[0];
    const results = await session.run({ [inputName]: inputTensor });
    const outputName = session.outputNames[0];
    const scores = results[outputName].data;
    const labels = await loadLabels();

    const maxIndex = scores.indexOf(Math.max(...scores));
    const predictedLabel = labels[maxIndex];
    const confidence = scores[maxIndex];

    // Display results
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
      <p><strong>Prediction:</strong> ${predictedLabel}</p>
      <p><strong>Confidence:</strong> ${(confidence * 100).toFixed(2)}%</p>
    `;
  } catch (error) {
    console.error('Error during inference:', error);
  }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  loadModel(); // Load the ONNX model when the page loads

  const imageInput = document.getElementById('imageInput');
  const runButton = document.getElementById('runInference');

  runButton.addEventListener('click', async () => {
    const imageFile = imageInput.files[0];
    if (imageFile) {
      await runInference(imageFile);
    } else {
      alert('Please upload an image first.');
    }
  });
});
