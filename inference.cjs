// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const fs = require('fs');
const ort = require('onnxruntime-node');
const path = require('path');
const imagePath = path.join(__dirname, 'test-image-2.jpg');
const labelsPath = path.join(__dirname, "resnet_labels.json")

const sharp = require('sharp');
const {Tensor} = require("onnxruntime-web");

// following code also works for onnxruntime-web.
const InferenceSession = ort.InferenceSession;
const targetSize = 224;
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];


function loadLabels(labelsPath) {
    const labelsData = fs.readFileSync(labelsPath, 'utf-8');
    return JSON.parse(labelsData); // Parse and return the labels as an array
}

async function extract_image(imagePath) {
    const image_buffer = await sharp(imagePath)
        .resize(targetSize, targetSize)
        .raw()
        .toBuffer();
    console.assert(image_buffer.length === targetSize*targetSize*3, "Extracted image leads to an invalid buffer size");
    return image_buffer;
}

// Function to preprocess the image
function preprocessImage(imageBuffer) {
    const floatImage = Float32Array.from(imageBuffer, (pixel) => pixel / 255.0);
    // Normalize channels:
    const normalizedImage = new Float32Array(floatImage.length);
    for (let i = 0; i < floatImage.length; i++) {
        const channel = i % 3; // Assuming RGB with 3 channels
        normalizedImage[i] = (floatImage[i] - mean[channel]) / std[channel];
    }
    //Rearrange dimensions to [C, H, W] from [H, W, C]
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

    // Add a batch dimension to create a tensor of shape [1, C, H, W]
    const tensor = new Tensor('float32', transposedImage, [1, 3, height, width]);
    return tensor;
}


async function getPrediction(session, imagePath, labelsPath) {
    try {
        const imageBuffer = await extract_image(imagePath);
        // Preprocess the image
        const inputTensor = preprocessImage(imageBuffer);
        // Get the model's input name
        const inputName = session.inputNames[0];
        // Run inference
        const results = await session.run({ [inputName]: inputTensor });
        // Extract the output tensor
        const outputName = session.outputNames[0];
        const scores = results[outputName].data; // Float32Array
        // Find the index of the highest value
        const maxIndex = scores.indexOf(Math.max(...scores));
        // Load class labels
        const labels = loadLabels(labelsPath);
        // Get the label for the highest index
        const predictedLabel = labels[maxIndex];

        return { classIndex: maxIndex, confidence: scores[maxIndex], label: predictedLabel };
    } catch (error) {
        console.error('Error during prediction:', error);
        throw error;
    }
}


// use an async context to call onnxruntime functions.
async function main() {
    try {
        const session01 = await InferenceSession.create('./onnx_model/resnet.onnx');
        const results = await getPrediction(session01, imagePath, labelsPath);
        console.log(results);
    } catch (e) {
        console.error(`failed to create inference session: ${e}`);
    }
    console.log("Completion")
}

main();
