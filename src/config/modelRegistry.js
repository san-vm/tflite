export const MODEL_REGISTRY = {
	mobileFaceNet: {
		id: 'mobileFaceNet',
		name: 'MobileFaceNet',
		fileName: 'MobileFaceNet.tflite',
		size: '10MB',
		description: 'Lightweight face recognition model optimized for mobile devices.',
		input: {
			size: 112,
			dtype: 'float32',
			normalization: 'zeroToOne',
			batchStrategy: 'duplicate',
			resizeMethod: 'bilinear',
			shape: [2, 112, 112, 3],
		},
		output: {
			type: 'embeddings',
		},
		documentation: [
			'Resizes the image to 112x112.',
			'Normalizes RGB pixels to float32 values in the [0, 1] range.',
			'Duplicates a single image so the final tensor matches the model batch requirement.',
		],
	},
	faceNet512: {
		id: 'faceNet512',
		name: 'FaceNet-512',
		fileName: 'facenet_512.tflite',
		size: '24MB',
		description: 'High-accuracy face recognition model with 512-dimensional embeddings.',
		input: {
			size: 160,
			dtype: 'float32',
			normalization: 'zeroToOne',
			batchStrategy: 'single',
			resizeMethod: 'bilinear',
			shape: [1, 160, 160, 3],
		},
		output: {
			type: 'embeddings',
		},
		documentation: [
			'Resizes the image to 160x160.',
			'Normalizes RGB pixels to float32 values in the [0, 1] range.',
			'Returns a single embedding vector per image.',
		],
	},
	mirnet_int8: {
		id: 'mirnet_int8',
		name: 'mirnet_int8',
		fileName: 'mirnet_int8.tflite',
		size: '24MB',
		description: 'Image enhancement model that returns an output image.',
		input: {
			size: 400,
			dtype: 'float32',
			normalization: 'zeroToOne',
			batchStrategy: 'single',
			resizeMethod: 'bilinear',
			shape: [1, 400, 400, 3],
		},
		output: {
			type: 'image',
		},
		documentation: [
			'Resizes the input image to 400x400.',
			'Normalizes RGB pixels to float32 values in the [0, 1] range.',
			'Interprets the model output as an RGB image and scales it back to the original preview size.',
		],
	},
	customVision256: {
		id: 'customVision256',
		name: 'Custom Vision 256',
		fileName: 'custom_vision_256.tflite',
		size: 'Add model file',
		description: 'Template entry for models that expect a [1, 256, 256, 3] float32 tensor.',
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
			'Place the model at public/models/custom_vision_256.tflite or change fileName in this registry.',
			'Preprocessing matches your Python snippet: resize to 256x256 with nearest-neighbor, divide by 255, keep 3 channels, then reshape to [1, 256, 256, 3].',
			'The app shows the raw JSON output by default because no postprocessing contract was provided for this model.',
		],
	},
};

export const MODEL_OPTIONS = Object.values(MODEL_REGISTRY);

export const DEFAULT_MODEL_ID = 'mobileFaceNet';

export const createInitialModelState = () => {
	const initialState = {};

	for (const model of MODEL_OPTIONS) {
		initialState[model.id] = null;
	}

	return initialState;
};

export const getModelConfig = (modelId) => MODEL_REGISTRY[modelId] ?? null;
