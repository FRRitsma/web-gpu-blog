import { InferenceSession, Tensor } from 'onnxruntime-web';


function resizeImage(imageFile) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        img.onload = () => {
            canvas.width = 224;
            canvas.height = 224;
            ctx.drawImage(img, 0, 0, 224, 224);
            const imageData = ctx.getImageData(0, 0, 224, 224).data;
            resolve(imageData);
        };

        img.onerror = (err) => reject(err);
        img.src = URL.createObjectURL(imageFile);
    });
}

async function main() {
    try{
        const session = await InferenceSession.create('./onnx_model/resnet.onnx');
        console.log(session);
    }
    catch(e){
        console.error("Failure")
    }
}


main();