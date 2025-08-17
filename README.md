# TFLite Model Playground

A fast, flexible, and privacy-first in-browser platform to test and run TensorFlow Lite (TFLite) models using WebAssembly. Built with React + Vite, this project supports real-time webcam inference, image enhancement, and model configuration — all running locally in the browser.

## ✨ Features

- 🔁 **General-Purpose TFLite Runner**: Easily load and run any compatible TFLite model
- 🎥 **Live Webcam Inference**: Perform real-time predictions on webcam frames
- 🌑 **Image Enhancement**: Includes a shadow removal model for improving low-light or dark images
- 🧠 **Face Embeddings**: Test face-related models like FaceNet or MobileNet for feature extraction
- 🛠️ **Custom Inference Pipeline**: Flexible structure to plug in your own preprocessing and postprocessing
- 🔐 **Fully Local**: No data leaves the browser — privacy by design
- ⚡ **Powered by WebAssembly**: Ultra-fast model execution using TFLite WASM backend

## 🔧 Technologies Used

- [Vite](https://vitejs.dev/)
- [React](https://reactjs.org/)
- [TensorFlow Lite Web](https://www.tensorflow.org/lite/guide/web)
- [WebAssembly](https://webassembly.org/)

## 📦 Model Support

Supports any TFLite model that is compatible with the WASM backend and meets browser memory limits.

Built-in examples:
- **FaceNet / MobileNet** – for facial feature extraction and embeddings
- **Shadow Remover** – for enhancing images taken in poor lighting conditions

> You can easily plug in your own `.tflite` model by modifying the config or loading via the UI (if enabled).

## 🖥️ How It Works

1. Load a model (from local assets or user input)
2. Input data via webcam or image
3. Run inference via TFLite WASM
4. Display the results or embeddings
5. Apply optional postprocessing (e.g., visualizations or enhancements)

## 🚀 Getting Started

### Prerequisites
- Node.js >= 16
- A modern browser (Chrome, Firefox, or Edge) with WASM + WebGL support
- Camera permissions (for webcam-based inference)

### Install and Run

```bash
git clone https://github.com/yourusername/tflite-model-playground.git
cd tflite-model-playground

npm install --force
npm run dev
