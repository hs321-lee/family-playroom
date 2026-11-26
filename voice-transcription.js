import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;

let transcriber;
self.onmessage = async ({data:samples}) => {
  try {
    transcriber ||= pipeline('automatic-speech-recognition','Xenova/whisper-tiny',{device:'wasm',dtype:'q8'});
    const recognise = await transcriber;
    const result = await recognise(samples,{language:'korean',task:'transcribe',return_timestamps:false});
    self.postMessage({text:result.text});
  } catch {
    transcriber = undefined;
    self.postMessage({error:true});
  }
};
