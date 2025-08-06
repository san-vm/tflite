import React, { useState, useRef } from 'react';
import { preprocessImage, preprocessSingleForMobileNet, runInference } from '../utils/modelUtils';

const ImageInput = ({ model, size }) => {
	const [selectedImage, setSelectedImage] = useState(null);
	const [imagePreview, setImagePreview] = useState(null);
	const [results, setResults] = useState(null);
	const [processing, setProcessing] = useState(false);
	const fileInputRef = useRef(null);
	const canvasRef = useRef(null);

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
			img.onload = async () => {
				try {
					// Preprocess image for the model
					let processedData = undefined
					if (size) {
						processedData = preprocessSingleForMobileNet(img, canvasRef.current);
					}
					else {
						processedData = preprocessImage(img, canvasRef.current);
					}

					// Run inference
					const inferenceResults = await runInference(model, processedData);

					// Display results
					setResults({
						success: true,
						embeddings: inferenceResults,
						timestamp: new Date().toISOString()
					});
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

	return (
		<div className="image-input">
			<h3>Image Input</h3>

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

			{results && (
				<div className="results-section">
					<h4>Processing Results:</h4>
					{results.success ? (
						<div className="success-results">
							<p>✅ Face processing completed successfully!</p>
							<p><strong>Embedding dimensions:</strong> {results.embeddings?.length || 'N/A'}</p>
							<p><strong>Processed at:</strong> {new Date(results.timestamp).toLocaleString()}</p>
							<details>
								<summary>View embedding data (first 10 values)</summary>
								<pre className="embedding-data">
									{JSON.stringify(results.embeddings?.slice(0, 10), null, 2)}
								</pre>
							</details>
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
