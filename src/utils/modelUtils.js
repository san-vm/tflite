import * as tf from '@tensorflow/tfjs';


// Preprocess one image to 112x112, returns tensor of shape [1,112,112,3]
export const preprocessImage = (img, canvas) => {
	const inputSize = 160;
	canvas.width = inputSize;
	canvas.height = inputSize;
	const ctx = canvas.getContext('2d');
	ctx.drawImage(img, 0, 0, inputSize, inputSize);

	// Get image data
	const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
	const pixels = imageData.data;

	// Convert to tensor and normalize
	const tensorData = new Float32Array(inputSize * inputSize * 3);

	for (let i = 0; i < pixels.length; i += 4) {
		const pixelIndex = i / 4;
		// Normalize pixel values to [0, 1] or [-1, 1] depending on model
		tensorData[pixelIndex * 3] = pixels[i] / 255.0;       // R
		tensorData[pixelIndex * 3 + 1] = pixels[i + 1] / 255.0; // G
		tensorData[pixelIndex * 3 + 2] = pixels[i + 2] / 255.0; // B
	}

	// Create tensor with shape [1, height, width, channels]
	const tensor = tf.tensor4d(tensorData, [1, inputSize, inputSize, 3]);

	return tensor;
};

// Preprocess one image to 112x112, returns tensor of shape [1,112,112,3]
export const preprocessImage112 = (img, canvas) => {
    const inputSize = 112;
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, inputSize, inputSize);

    const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
    const pixels = imageData.data;
    const tensorData = new Float32Array(inputSize * inputSize * 3);

    for (let i = 0; i < pixels.length; i += 4) {
        const pixelIndex = i / 4;
        tensorData[pixelIndex * 3] = pixels[i] / 255.0;
        tensorData[pixelIndex * 3 + 1] = pixels[i + 1] / 255.0;
        tensorData[pixelIndex * 3 + 2] = pixels[i + 2] / 255.0;
    }
    // Shape: [1, 112, 112, 3]
    return tf.tensor4d(tensorData, [1, inputSize, inputSize, 3]);
};

// Combine two preprocessed tensors into a batch of shape [2,112,112,3]
export const preprocessBatchForMobileNet = (img1, img2, canvas) => {
    const tensor1 = preprocessImage112(img1, canvas); // shape [1,112,112,3]
    const tensor2 = preprocessImage112(img2, canvas); // shape [1,112,112,3]
    // Concatenate along 0th axis (batch)
    const batchTensor = tf.concat([tensor1, tensor2], 0); // shape [2,112,112,3]
    tensor1.dispose();
    tensor2.dispose();
    return batchTensor;
};

// If you only have one image and must duplicate for batch size 2:
export const preprocessSingleForMobileNet = (img, canvas) => {
    const tensor = preprocessImage112(img, canvas); // shape [1,112,112,3]
    const batchTensor = tf.concat([tensor, tensor], 0); // shape [2,112,112,3]
    tensor.dispose();
    return batchTensor;
};


/**
 * Run inference on the TFLite model
 * @param {Object} model - Loaded TFLite model
 * @param {tf.Tensor} inputTensor - Preprocessed input tensor
 * @returns {Array} - Model output (face embeddings)
 */
export const runInference = async (model, inputTensor) => {
	if (!model) {
		throw new Error('Model not loaded');
	}

	try {
		// Run prediction
		const prediction = await model.predict(inputTensor);

		// Convert tensor to array
		let outputData;
		if (prediction.arraySync) {
			outputData = prediction.arraySync();
		} else {
			outputData = await prediction.data();
		}

		// Clean up tensors
		inputTensor.dispose();
		prediction.dispose();

		// Return flattened array (face embeddings)
		return Array.isArray(outputData[0]) ? outputData[0] : Array.from(outputData);

	} catch (error) {
		// Clean up tensor in case of error
		inputTensor.dispose();
		throw new Error(`Inference failed: ${error.message}`);
	}
};

/**
 * Calculate cosine similarity between two face embeddings
 * @param {Array} embedding1 - First face embedding
 * @param {Array} embedding2 - Second face embedding
 * @returns {number} - Similarity score (0-1)
 */
export const calculateSimilarity = (embedding1, embedding2) => {
	if (embedding1.length !== embedding2.length) {
		throw new Error('Embeddings must have the same length');
	}

	// Calculate dot product
	let dotProduct = 0;
	let norm1 = 0;
	let norm2 = 0;

	for (let i = 0; i < embedding1.length; i++) {
		dotProduct += embedding1[i] * embedding2[i];
		norm1 += embedding1[i] * embedding1[i];
		norm2 += embedding2[i] * embedding2[i];
	}

	// Calculate cosine similarity
	const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

	// Convert to 0-1 range
	return (similarity + 1) / 2;
};

/**
 * Normalize face embedding vector
 * @param {Array} embedding - Face embedding vector
 * @returns {Array} - L2 normalized embedding
 */
export const normalizeEmbedding = (embedding) => {
	const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
	return embedding.map(val => val / norm);
};
