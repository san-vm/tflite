# TFLite Model Playground

React + Vite app for loading TensorFlow Lite models in the browser with the `@tensorflow/tfjs-tflite` WASM runtime.

## What changed

The project now uses a central model registry at `src/config/modelRegistry.js`.

Each model entry defines:

- The `.tflite` filename under `public/models`
- Input shape and resize size
- Input dtype and normalization
- Resize behavior such as nearest-neighbor vs bilinear
- Output mode (`embeddings`, `image`, or `generic`)
- Short usage notes shown in the UI

This makes it possible to add new models without duplicating `if/else` logic in the image and webcam components.

## Built-in models

- `MobileFaceNet`
- `FaceNet-512`
- `mirnet_int8`
- `Custom Vision 256`

`Custom Vision 256` is the new template entry for the preprocessing contract you shared.

## New 256x256 model contract

The `customVision256` registry entry is configured to match this preprocessing flow:

```python
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
img = np.asarray(np.float32(img / 255))
img = np.reshape(img, (1, 256, 256, 3))
```

In the browser app that means:

- Resize input images to `256 x 256`
- Use nearest-neighbor resizing
- Convert pixel values to `float32`
- Normalize pixels to the `[0, 1]` range
- Produce a tensor shaped `[1, 256, 256, 3]`

## How to add your model

1. Copy your `.tflite` file into `public/models/`.
2. Open `src/config/modelRegistry.js`.
3. Update the `customVision256.fileName` value to match your actual filename, or add another registry entry if you want a separate model option in the UI.
4. If the model output is not raw JSON, update `output.type`:
   - `embeddings` for a 1D embedding vector
   - `image` for an RGB image output
   - `generic` for raw object/array output
5. Start the app and load the model from the dropdown.

## Example registry entry

```js
customVision256: {
  id: 'customVision256',
  name: 'Custom Vision 256',
  fileName: 'your_model_name.tflite',
  size: 'Add model file',
  description: 'Model expecting a [1, 256, 256, 3] float32 input tensor.',
  input: {
    size: 256,
    dtype: 'float32',
    normalization: 'zeroToOne',
    batchStrategy: 'single',
    resizeMethod: 'nearest',
    shape: [1, 256, 256, 3],
  },
  output: {
    type: 'generic',
  },
  documentation: [
    'Resize to 256x256 with nearest-neighbor interpolation.',
    'Normalize to float32 in the [0, 1] range.',
    'Return the raw model output until a postprocessing step is defined.',
  ],
}
```

## Output handling

The app now supports three output modes:

- `embeddings`: the first embedding vector is extracted and previewed
- `image`: the returned tensor is rendered to a canvas and can be saved as PNG
- `generic`: the raw prediction is shown as formatted JSON

If your new model needs custom postprocessing beyond that, extend `runInference` or add a model-specific postprocessing step in `src/utils/modelUtils.js`.

## Development

### Install

```bash
npm install --force
```

### Run

```bash
npm run dev
```

### Build

```bash
npm run build
```
