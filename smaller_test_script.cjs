// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const fs = require('fs');
const util = require('util');
const ort = require('onnxruntime-node');
const path = require('path');
const imagePath = path.join(__dirname, 'test-image-3.jpg');
const labelsPath = path.join(__dirname, "resnet_labels.json")

const sharp = require('sharp');

// following code also works for onnxruntime-web.
const InferenceSession = ort.InferenceSession;


//
function loadLabels(labelsPath) {
    const labelsData = fs.readFileSync(labelsPath, 'utf-8');
    return JSON.parse(labelsData); // Parse and return the labels as an array
}

// Function to preprocess the image
async function preprocessImage(imagePath, targetSize) {
    // Load and resize the image to the target dimensions
    const image = await sharp(imagePath)
        .resize(targetSize, targetSize)
        .raw()
        .toBuffer();

    // Normalize the pixel values to [0, 1] and convert to Float32Array
    let floatImage = Float32Array.from(image, (pixel) => pixel / 255.0);

    // Step 4: Normalize using mean and std for each channel
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const normalizedImage = new Float32Array(floatImage.length);

    // Rearrange data to [C, H, W]
    const numChannels = 3;
    for (let i = 0; i < floatImage.length; i++) {
        const channel = i % numChannels;
        floatImage[i] = (floatImage[i] - mean[channel]) / std[channel];
    }

    // Save the floatImage as an image
    const scaledImage = Uint8Array.from(floatImage.map((pixel) => Math.round(pixel * 255))); // Scale back to 0-255
    await sharp(Buffer.from(scaledImage), {
        raw: {
            width: targetSize,
            height: targetSize,
            channels: 3, // Assuming RGB
        },
    })
        .toFile('output-image.png'); // Save as PNG
    console.log('Saved floatImage as output-image.png');


    // Create a tensor with shape [1, 3, targetSize, targetSize] (batch size = 1, 3 channels)
    const tensor = new ort.Tensor('float32', floatImage, [1, 3, targetSize, targetSize]);
    return tensor;
}

async function getPrediction(session, imagePath, labelsPath) {
    try {
        // Preprocess the image
        const inputTensor = await preprocessImage(imagePath, 224);

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

    // let tensor =
    // console.log(tensor);

    try {
        // create session option object
        const options = createMySessionOptions();

        //
        // create inference session from a ONNX model file path or URL
        //
        const session01 = await InferenceSession.create('./onnx_model/resnet.onnx');
        const session01_B = await InferenceSession.create('./onnx_model/resnet.onnx', options); // specify options

        //
        // create inference session from an Node.js Buffer (Uint8Array)
        //
        const buffer02 = await readMyModelDataFile('./onnx_model/resnet.onnx'); // buffer is Uint8Array
        const session02 = await InferenceSession.create(buffer02);
        const session02_B = await InferenceSession.create(buffer02, options); // specify options

        //
        // create inference session from an ArrayBuffer
        //
        const arrayBuffer03 = buffer02.buffer;
        const offset03 = buffer02.byteOffset;
        const length03 = buffer02.byteLength;
        const session03 = await InferenceSession.create(arrayBuffer03, offset03, length03);
        const session03_B = await InferenceSession.create(arrayBuffer03, offset03, length03, options); // specify options

        // example for browser
        //const arrayBuffer03_C = await fetchMyModel('./model.onnx');
        //const session03_C = await InferenceSession.create(arrayBuffer03_C);


        // const inputTensor = await preprocessImage(imagePath, 224);
        // const inputName = session01.inputNames[0]; // Most models have a single input
        // const results = await session01.run({ [inputName]: inputTensor });
        const results = await getPrediction(session01, imagePath, labelsPath);
        console.log(results);

    } catch (e) {
        console.error(`failed to create inference session: ${e}`);
    }


    console.log("Completion")
}

main();

function createMySessionOptions() {
    // session options: please refer to the other example for details usage for session options

    // example of a session option object in node.js:
    // specify intra operator threads number to 1 and disable CPU memory arena
    return { intraOpNumThreads: 1, enableCpuMemArena: false }

    // example of a session option object in browser:
    // specify WebAssembly exection provider
    //return { executionProviders: ['wasm'] };

}

async function readMyModelDataFile(filepathOrUri) {
    // read model file content (Node.js) as Buffer (Uint8Array)
    return await util.promisify(fs.readFile)(filepathOrUri);
}

async function fetchMyModel(filepathOrUri) {
    // use fetch to read model file (browser) as ArrayBuffer
    if (typeof fetch !== 'undefined') {
        const response = await fetch(filepathOrUri);
        return await response.arrayBuffer();
    }
}