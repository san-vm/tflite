import * as tf from '@tensorflow/tfjs';

const isTensorLike = (value) =>
	Boolean(value) &&
	typeof value.array === 'function' &&
	typeof value.dispose === 'function';

const configureCanvasResize = (ctx, resizeMethod = 'bilinear') => {
	const useNearestNeighbor = resizeMethod === 'nearest';
	ctx.imageSmoothingEnabled = !useNearestNeighbor;

	if ('imageSmoothingQuality' in ctx) {
		ctx.imageSmoothingQuality = useNearestNeighbor ? 'low' : 'high';
	}
};

const drawImageToCanvas = (img, canvas, size, resizeMethod = 'bilinear') => {
	canvas.width = size;
	canvas.height = size;

	const ctx = canvas.getContext('2d');
	configureCanvasResize(ctx, resizeMethod);
	ctx.clearRect(0, 0, size, size);
	ctx.drawImage(img, 0, 0, size, size);

	return ctx;
};

export const preprocessImage = (img, canvas, options = {}) => {
	const normalizedOptions =
		typeof options === 'number'
			? { size: options }
			: options;
	const {
		size = 160,
		dtype = 'float32',
		normalization = 'zeroToOne',
		resizeMethod = 'bilinear',
	} = normalizedOptions;

	const ctx = drawImageToCanvas(img, canvas, size, resizeMethod);
	const imageData = ctx.getImageData(0, 0, size, size);
	const pixels = imageData.data;
	const tensorData =
		dtype === 'int32'
			? new Int32Array(size * size * 3)
			: new Float32Array(size * size * 3);

	for (let i = 0; i < pixels.length; i += 4) {
		const pixelIndex = i / 4;
		const red = pixels[i];
		const green = pixels[i + 1];
		const blue = pixels[i + 2];

		if (dtype === 'int32' || normalization === 'none') {
			tensorData[pixelIndex * 3] = red;
			tensorData[pixelIndex * 3 + 1] = green;
			tensorData[pixelIndex * 3 + 2] = blue;
		}
		else {
			tensorData[pixelIndex * 3] = red / 255;
			tensorData[pixelIndex * 3 + 1] = green / 255;
			tensorData[pixelIndex * 3 + 2] = blue / 255;
		}
	}

	return tf.tensor4d(tensorData, [1, size, size, 3], dtype);
};

export const preprocessImageUint8 = (img, canvas, size = 160) =>
	preprocessImage(img, canvas, {
		size,
		dtype: 'int32',
		normalization: 'none',
	});

export const preprocessBatchForMobileNet = (img1, img2, canvas, size = 112) => {
	const tensor1 = preprocessImage(img1, canvas, size);
	const tensor2 = preprocessImage(img2, canvas, size);
	const batchTensor = tf.concat([tensor1, tensor2], 0);

	tensor1.dispose();
	tensor2.dispose();

	return batchTensor;
};

export const preprocessSingleForMobileNet = (img, canvas, size = 112) => {
	const tensor = preprocessImage(img, canvas, size);
	const batchTensor = tf.concat([tensor, tensor], 0);

	tensor.dispose();

	return batchTensor;
};

export const prepareModelInput = (img, canvas, modelConfig) => {
	if (!modelConfig?.input) {
		throw new Error('Missing model input configuration');
	}

	const {
		size,
		dtype = 'float32',
		normalization = 'zeroToOne',
		batchStrategy = 'single',
		resizeMethod = 'bilinear',
	} = modelConfig.input;

	const inputTensor = preprocessImage(img, canvas, {
		size,
		dtype,
		normalization,
		resizeMethod,
	});

	if (batchStrategy === 'duplicate') {
		const batchTensor = tf.concat([inputTensor, inputTensor], 0);
		inputTensor.dispose();
		return batchTensor;
	}

	return inputTensor;
};

const serializePrediction = async (prediction) => {
	if (isTensorLike(prediction)) {
		const serializedTensor = await prediction.array();
		prediction.dispose();
		return serializedTensor;
	}

	if (Array.isArray(prediction)) {
		return Promise.all(prediction.map((item) => serializePrediction(item)));
	}

	if (prediction && typeof prediction === 'object') {
		const entries = await Promise.all(
			Object.entries(prediction).map(async ([key, value]) => [
				key,
				await serializePrediction(value),
			])
		);

		return Object.fromEntries(entries);
	}

	return prediction;
};

const getPrimaryOutput = (output) => {
	if (Array.isArray(output)) {
		return output;
	}

	if (output && typeof output === 'object') {
		const values = Object.values(output);
		if (values.length === 1) {
			return values[0];
		}
	}

	return output;
};

const extractEmbeddingOutput = (output) => {
	const primaryOutput = getPrimaryOutput(output);

	if (Array.isArray(primaryOutput) && primaryOutput.length > 0) {
		if (
			Array.isArray(primaryOutput[0]) &&
			typeof primaryOutput[0][0] === 'number'
		) {
			return primaryOutput[0];
		}

		if (typeof primaryOutput[0] === 'number') {
			return primaryOutput;
		}
	}

	throw new Error('Unable to interpret model output as an embedding vector');
};

const extractImageOutput = (output) => {
	const primaryOutput = getPrimaryOutput(output);

	if (
		Array.isArray(primaryOutput) &&
		primaryOutput.length === 1 &&
		Array.isArray(primaryOutput[0])
	) {
		return primaryOutput[0];
	}

	return primaryOutput;
};

const formatModelOutput = (output, modelConfig) => {
	const outputType = modelConfig?.output?.type ?? 'generic';

	if (outputType === 'embeddings') {
		return extractEmbeddingOutput(output);
	}

	if (outputType === 'image') {
		return extractImageOutput(output);
	}

	return output;
};

export const runInference = async (model, inputTensor, modelConfig) => {
	if (!model) {
		throw new Error('Model not loaded');
	}

	let inputDisposed = false;

	try {
		const prediction = await model.predict(inputTensor);
		inputTensor.dispose();
		inputDisposed = true;

		const serializedOutput = await serializePrediction(prediction);
		return formatModelOutput(serializedOutput, modelConfig);
	}
	catch (error) {
		if (!inputDisposed) {
			inputTensor.dispose();
		}
		throw new Error(`Inference failed: ${error.message}`);
	}
};

export const renderImageOutputToCanvas = (
	rawData,
	canvas,
	targetWidth,
	targetHeight
) => {
	const height = rawData?.length ?? 0;
	const width = rawData?.[0]?.length ?? 0;

	if (!width || !height) {
		throw new Error('Invalid output image dimensions');
	}

	const rgbaData = new Uint8ClampedArray(width * height * 4);
	let index = 0;

	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			const pixel = rawData[y][x] ?? [0, 0, 0];

			rgbaData[index] = Math.max(0, Math.min(255, Math.round((pixel[0] ?? 0) * 255)));
			rgbaData[index + 1] = Math.max(0, Math.min(255, Math.round((pixel[1] ?? 0) * 255)));
			rgbaData[index + 2] = Math.max(0, Math.min(255, Math.round((pixel[2] ?? 0) * 255)));
			rgbaData[index + 3] = 255;
			index += 4;
		}
	}

	const imageData = new ImageData(rgbaData, width, height);
	const outputCanvas = canvas;
	outputCanvas.width = targetWidth ?? width;
	outputCanvas.height = targetHeight ?? height;

	const outputContext = outputCanvas.getContext('2d');
	const tempCanvas = document.createElement('canvas');
	tempCanvas.width = width;
	tempCanvas.height = height;

	const tempContext = tempCanvas.getContext('2d');
	tempContext.putImageData(imageData, 0, 0);

	outputContext.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
	outputContext.drawImage(
		tempCanvas,
		0,
		0,
		width,
		height,
		0,
		0,
		outputCanvas.width,
		outputCanvas.height
	);

	return { width, height };
};

export const calculateSimilarity = (embedding1, embedding2) => {
	if (embedding1.length !== embedding2.length) {
		throw new Error('Embeddings must have the same length');
	}

	let dotProduct = 0;
	let norm1 = 0;
	let norm2 = 0;

	for (let i = 0; i < embedding1.length; i += 1) {
		dotProduct += embedding1[i] * embedding2[i];
		norm1 += embedding1[i] * embedding1[i];
		norm2 += embedding2[i] * embedding2[i];
	}

	const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

	return (similarity + 1) / 2;
};

export const normalizeEmbedding = (embedding) => {
	const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
	return embedding.map((value) => value / norm);
};
