import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
const InferenceSession = ort.InferenceSession;
async function main(){
    const session = await InferenceSession.create('./onnx_model/resnet.onnx')
}

main();
