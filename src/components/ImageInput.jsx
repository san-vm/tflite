import React, { useState, useRef } from 'react';
import { preprocessImage, preprocessSingleForMobileNet, runInference } from '../utils/modelUtils';

const modelInfo = {
	mobileFaceNet: {
		name: 'MobileFaceNet',
		size: '10MB',
		description: 'Lightweight face recognition model optimized for mobile devices',
		outputType: 'embeddings', // embedding output
	},
	faceNet512: {
		name: 'FaceNet-512',
		size: '24MB',
		description: 'High-accuracy face recognition with 512-dimensional embeddings',
		outputType: 'embeddings',
	},
	mirnet_int8: {
		name: 'mirnet_int8',
		size: '24MB',
		description: 'Image enhancement model producing image output',
		outputType: 'image', // image output
	},
};

const ImageInput = ({ model, modelName }) => {
	const [selectedImage, setSelectedImage] = useState(null);
	const [imagePreview, setImagePreview] = useState(null);
	const [results, setResults] = useState(null);
	const [processing, setProcessing] = useState(false);
	const fileInputRef = useRef(null);
	const canvasRef = useRef(null);
	const outputCanvasRef = useRef(null);

	const modelDetails = modelInfo[modelName] || {};

	const handleImageSelect = (event) => {
		const file = event.target.files[0];
		if (file && file.type.startsWith('image/')) {
			setSelectedImage(file);

			// Create preview
			const reader = new FileReader();
			reader.onload = (e) => {
				setImagePreview(e.target.result);
			};
			reader.readAsDataURL(file);

			// Clear previous results
			setResults(null);
		}
	};

	const processImage = async () => {
		if (!selectedImage || !model) {
			alert('Please select an image and load a model first');
			return;
		}

		setProcessing(true);
		setResults(null);

		try {
			// Create image element from file
			const img = new Image();
			img.crossOrigin = "anonymous"; // just in case
			img.onload = async () => {
				const origWidth = img.width;
				const origHeight = img.height;
				try {
					let processedData;
					if (modelName === "mobileFaceNet") {
						processedData = preprocessSingleForMobileNet(img, canvasRef.current);
					}
					else if (modelName === "faceNet512") {
						processedData = preprocessImage(img, canvasRef.current);
					}
					else if (modelName === "mirnet_int8") {
						processedData = preprocessImage(img, canvasRef.current, 400);
					}

					// Measure inference time
					const startTime = performance.now();
					const inferenceResults = await runInference(model, processedData);
					const endTime = performance.now();
					const inferenceTimeMs = (endTime - startTime).toFixed(2);

					// Handle output display based on outputType
					if (modelDetails.outputType === 'image') {
						const rawData = inferenceResults; // [[[r, g, b], ...], ...]

						// Dynamically get output dimensions from model output
						const height = rawData.length;
						const width = rawData[0]?.length || 0;

						if (!width || !height) {
							throw new Error('Invalid output image dimensions');
						}

						// Flatten to RGBA
						const rgbaData = new Uint8ClampedArray(width * height * 4);
						let idx = 0;
						for (let y = 0; y < height; y++) {
							for (let x = 0; x < width; x++) {
								const pixel = rawData[y][x]; // [r, g, b]

								rgbaData[idx++] = Math.round(pixel[0] * 255); // R
								rgbaData[idx++] = Math.round(pixel[1] * 255); // G
								rgbaData[idx++] = Math.round(pixel[2] * 255); // B
								rgbaData[idx++] = 255;                        // A
							}
						}

						const imageData = new ImageData(rgbaData, width, height);

						// origWidth and origHeight come from the original loaded image dimensions
						outputCanvasRef.current.width = origWidth;
						outputCanvasRef.current.height = origHeight;

						const ctx = outputCanvasRef.current.getContext('2d');

						// Create a temporary canvas to hold the raw output image data
						const tempCanvas = document.createElement('canvas');
						tempCanvas.width = width;
						tempCanvas.height = height;
						const tempCtx = tempCanvas.getContext('2d');
						tempCtx.putImageData(imageData, 0, 0);

						// Draw the temp canvas scaled to original image size
						ctx.clearRect(0, 0, origWidth, origHeight);
						ctx.drawImage(tempCanvas, 0, 0, width, height, 0, 0, origWidth, origHeight);

						setResults({
							success: true,
							imageOutput: true,
							embeddings: inferenceResults,
							inferenceTimeMs,
							timestamp: new Date().toISOString(),
						});
					}

					else {
						// Embedding output
						setResults({
							success: true,
							embeddings: inferenceResults,
							inferenceTimeMs,
							timestamp: new Date().toISOString(),
						});
					}

				} catch (err) {
					console.error('Processing error:', err);
					setResults({
						success: false,
						error: err.message
					});
				} finally {
					setProcessing(false);
				}
			};
			img.src = imagePreview;
		} catch (err) {
			console.error('Image loading error:', err);
			setResults({
				success: false,
				error: err.message
			});
			setProcessing(false);
		}
	};

	const clearImage = () => {
		setSelectedImage(null);
		setImagePreview(null);
		setResults(null);
		if (fileInputRef.current) {
			fileInputRef.current.value = '';
		}
	};

	const saveOutputImage = () => {
		if (!outputCanvasRef.current) return;

		const link = document.createElement('a');
		link.download = `${modelName}_output_${Date.now()}.png`;
		link.href = outputCanvasRef.current.toDataURL('image/png');
		link.click();
	};

	return (
		<div className="image-input">
			<h3>Image Input - {modelDetails.name || modelName}</h3>
			<p><strong>Model size:</strong> {modelDetails.size || 'N/A'}</p>
			<p><strong>Description:</strong> {modelDetails.description || 'No description available.'}</p>

			<div className="file-input-section">
				<input
					type="file"
					accept="image/*"
					onChange={handleImageSelect}
					ref={fileInputRef}
					className="file-input"
				/>

				<div className="input-buttons">
					<button
						onClick={() => fileInputRef.current?.click()}
						className="select-button"
					>
						Select Image
					</button>

					{selectedImage && (
						<button onClick={clearImage} className="clear-button">
							Clear
						</button>
					)}
				</div>
			</div>

			{imagePreview && (
				<div className="image-preview">
					<h4>Selected Image:</h4>
					<img
						src={imagePreview}
						alt="Selected"
						className="preview-image"
					/>

					<button
						onClick={processImage}
						disabled={processing || !model}
						className="process-button"
					>
						{processing ? 'Processing...' : 'Process Image'}
					</button>
				</div>
			)}

			<canvas ref={canvasRef} style={{ display: 'none' }} />

			{modelDetails.outputType === 'image' && (
				<div style={{ marginTop: '20px' }}>
					<h4>Output Image:</h4>
					<canvas ref={outputCanvasRef} style={{ border: '1px solid #ccc', maxWidth: '50%' }} />
					{results?.success && (
						<button onClick={saveOutputImage} style={{ marginTop: '10px' }}>
							Save Image
						</button>
					)}
				</div>
			)}

			{results && (
				<div className="results-section" style={{ marginTop: '20px' }}>
					<h4>Processing Results:</h4>
					{results.success ? (
						<div className="success-results">
							<p>✅ Face processing completed successfully!</p>
							<p><strong>Embedding dimensions:</strong> {results.embeddings?.length || 'N/A'}</p>
							<details>
								<summary>View embedding data (first 10 values)</summary>
								<pre className="embedding-data">
									{JSON.stringify(results.embeddings?.slice(0, 10), null, 2)}
								</pre>
							</details>
							<p><strong>Inference time:</strong> {(results.inferenceTimeMs / 1000).toFixed(2)} s</p>
							<p><strong>Processed at:</strong> {new Date(results.timestamp).toLocaleString()}</p>
						</div>
					) : (
						<div className="error-results">
							<p>❌ Processing failed: {results.error}</p>
						</div>
					)}
				</div>
			)}
		</div>
	);
};

export default ImageInput;
