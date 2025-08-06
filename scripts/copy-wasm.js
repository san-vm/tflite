import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create public/wasm directory if it doesn't exist
const wasmDir = path.join(__dirname, '..', 'public', 'wasm');
if (!fs.existsSync(wasmDir)) {
  fs.mkdirSync(wasmDir, { recursive: true });
}

// Copy TensorFlow.js backend wasm files
const tfjsBackendWasmDir = path.join(__dirname, '..', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
const tfjsBackendWasmFiles = [
  'tfjs-backend-wasm.wasm',
  'tfjs-backend-wasm-simd.wasm',
  'tfjs-backend-wasm-threaded-simd.wasm'
];

// Copy TensorFlow Lite wasm files and clients
const tfliteWasmDir = path.join(__dirname, '..', 'node_modules', '@tensorflow', 'tfjs-tflite', 'dist');
const tfliteWasmFiles = [
  'tflite_web_api_cc.js',
  'tflite_web_api_cc.wasm',
  'tflite_web_api_cc_simd.js',
  'tflite_web_api_cc_simd.wasm',
  'tflite_web_api_cc_simd_threaded.js',
  'tflite_web_api_cc_simd_threaded.wasm',
  'tflite_web_api_cc_simd_threaded.worker.js',
  'tflite_web_api_cc_threaded.js',
  'tflite_web_api_cc_threaded.wasm',
  'tflite_web_api_cc_threaded.worker.js',
  'tflite_web_api_client.js'
];

const copyFiles = (sourceDir, files) => {
  files.forEach(file => {
    const source = path.join(sourceDir, file);
    const dest = path.join(wasmDir, file);

    if (fs.existsSync(source)) {
      fs.copyFileSync(source, dest);
      console.log(`Copied ${file} to public/wasm/`);
    } else {
      console.warn(`Warning: File ${file} not found in ${sourceDir}`);
    }
  });
};

// Copy both sets of files
copyFiles(tfjsBackendWasmDir, tfjsBackendWasmFiles);
copyFiles(tfliteWasmDir, tfliteWasmFiles);

console.log('All WASM files setup complete!');
