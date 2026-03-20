import React, { useRef, useState } from 'react';
import { getModelConfig } from '../config/modelRegistry';
import {
	prepareModelInput,
	renderImageOutputToCanvas,
	runInference,
} from '../utils/modelUtils';

const getPreviewPayload = (output) => {
	if (Array.isArray(output)) {
		if (typeof output[0] === 'number') {
			return output.slice(0, 10);
		}

		if (Array.isArray(output[0])) {
			return output.slice(0, 2);
		}
	}

	return output;
};

const ResultSummary = ({ results, modelDetails }) => {
	if (!results) {
		return null;
	}

	const outputType = modelDetails.output.type;

	if (!results.success) {
		return (
			<div className="error-results">
				<p>Processing failed: {results.error}</p>
			</div>
		);
	}

	if (outputType === 'image') {
		return (
			<div className="success-results">
				<p>Image processing completed successfully.</p>
				<p><strong>Output resolution:</strong> {results.outputDimensions?.width} x {results.outputDimensions?.height}</p>
				<p><strong>Inference time:</strong> {(results.inferenceTimeMs / 1000).toFixed(2)} s</p>
				<p><strong>Processed at:</strong> {new Date(results.timestamp).toLocaleString()}</p>
			</div>
		);
	}

	if (outputType === 'embeddings') {
		return (
			<div className="success-results">
				<p>Embedding extraction completed successfully.</p>
				<p><strong>Embedding dimensions:</strong> {results.output?.length ?? 'N/A'}</p>
				<details>
					<summary>View embedding data (first 10 values)</summary>
					<pre className="embedding-data">
						{JSON.stringify(getPreviewPayload(results.output), null, 2)}
					</pre>
				</details>
				<p><strong>Inference time:</strong> {(results.inferenceTimeMs / 1000).toFixed(2)} s</p>
				<p><strong>Processed at:</strong> {new Date(results.timestamp).toLocaleString()}</p>
			</div>
		);
	}

	return (
		<div className="success-results">
			<p>Inference completed successfully.</p>
			<p><strong>Response format:</strong> Raw model output</p>
			<details>
				<summary>View output JSON</summary>
				<pre className="embedding-data">
					{JSON.stringify(results.output, null, 2)}
				</pre>
			</details>
			<p><strong>Inference time:</strong> {(results.inferenceTimeMs / 1000).toFixed(2)} s</p>
			<p><strong>Processed at:</strong> {new Date(results.timestamp).toLocaleString()}</p>
		</div>
	);
};

const ImageInput = ({ model, modelName }) => {
	const [selectedImage, setSelectedImage] = useState(null);
	const [imagePreview, setImagePreview] = useState(null);
	const [results, setResults] = useState(null);
	const [processing, setProcessing] = useState(false);
	const fileInputRef = useRef(null);
	const canvasRef = useRef(null);
	const outputCanvasRef = useRef(null);

	const modelDetails = getModelConfig(modelName);

	const handleImageSelect = (event) => {
		const file = event.target.files[0];
		if (!file || !file.type.startsWith('image/')) {
			return;
		}

		setSelectedImage(file);

		const reader = new FileReader();
		reader.onload = (loadEvent) => {
			setImagePreview(loadEvent.target.result);
		};
		reader.readAsDataURL(file);

		setResults(null);
	};

	const processImage = async () => {
		if (!selectedImage || !model || !modelDetails) {
			alert('Please select an image and load a model first');
			return;
		}

		setProcessing(true);
		setResults(null);

		try {
			const img = new Image();
			img.crossOrigin = 'anonymous';
			img.onload = async () => {
				try {
					const processedData = prepareModelInput(img, canvasRef.current, modelDetails);
					const startTime = performance.now();
					const inferenceResults = await runInference(model, processedData, modelDetails);
					const endTime = performance.now();
					const inferenceTimeMs = endTime - startTime;

					if (modelDetails.output.type === 'image') {
						const outputDimensions = renderImageOutputToCanvas(
							inferenceResults,
							outputCanvasRef.current,
							img.width,
							img.height
						);

						setResults({
							success: true,
							output: inferenceResults,
							outputDimensions,
							inferenceTimeMs,
							timestamp: new Date().toISOString(),
						});
					}
					else {
						setResults({
							success: true,
							output: inferenceResults,
							inferenceTimeMs,
							timestamp: new Date().toISOString(),
						});
					}
				}
				catch (err) {
					console.error('Processing error:', err);
					setResults({
						success: false,
						error: err.message,
					});
				}
				finally {
					setProcessing(false);
				}
			};
			img.src = imagePreview;
		}
		catch (err) {
			console.error('Image loading error:', err);
			setResults({
				success: false,
				error: err.message,
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
		if (!outputCanvasRef.current) {
			return;
		}

		const link = document.createElement('a');
		link.download = `${modelName}_output_${Date.now()}.png`;
		link.href = outputCanvasRef.current.toDataURL('image/png');
		link.click();
	};

	if (!modelDetails) {
		return null;
	}

	return (
		<div className="image-input">
			<div className="card-header">
				<h2>Image Runner</h2>
				<p>{modelDetails.name} · {modelDetails.description}</p>
			</div>
			<p className="section-meta"><strong>Model size:</strong> {modelDetails.size}</p>

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
					<h4>Selected Image</h4>
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

			{modelDetails.output.type === 'image' && (
				<div className="output-image-section">
					<h4>Output Image</h4>
					<canvas ref={outputCanvasRef} className="output-canvas" />
					{results?.success && (
						<button onClick={saveOutputImage} className="process-button">
							Save Image
						</button>
					)}
				</div>
			)}

			{results && (
				<div className="results-section">
					<h4>Processing Results</h4>
					<ResultSummary results={results} modelDetails={modelDetails} />
				</div>
			)}
		</div>
	);
};

export default ImageInput;
