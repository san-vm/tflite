import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { preprocessImage, preprocessSingleForMobileNet, runInference } from '../utils/modelUtils';

const WebcamCapture = ({ model, size }) => {
	const [isWebcamActive, setIsWebcamActive] = useState(false);
	const [capturedImage, setCapturedImage] = useState(null);
	const [results, setResults] = useState(null);
	const [processing, setProcessing] = useState(false);
	const [realTimeMode, setRealTimeMode] = useState(false);
	const [realTimeResults, setRealTimeResults] = useState(null);

	const webcamRef = useRef(null);
	const canvasRef = useRef(null);
	const intervalRef = useRef(null);

	const videoConstraints = {
		width: 640,
		height: 480,
		facingMode: "user"
	};

	const startWebcam = useCallback(() => {
		setIsWebcamActive(true);
		setCapturedImage(null);
		setResults(null);
	}, []);

	const stopWebcam = useCallback(() => {
		setIsWebcamActive(false);
		setRealTimeMode(false);
		if (intervalRef.current) {
			clearInterval(intervalRef.current);
		}
	}, []);

	const capturePhoto = useCallback(() => {
		const imageSrc = webcamRef.current.getScreenshot();
		setCapturedImage(imageSrc);
		setResults(null);
	}, []);

	const processCapture = async (ignoreErr = false) => {
		if ((!capturedImage || !model) && !ignoreErr) {
			alert('Please capture an image and load a model first');
			return;
		}

		setProcessing(true);
		setResults(null);

		try {
			const img = new Image();
			img.onload = async () => {
				try {
					let processedData = undefined
					if (size) {
						processedData = preprocessSingleForMobileNet(img, canvasRef.current);
					}
					else {
						processedData = preprocessImage(img, canvasRef.current);
					}

					const inferenceResults = await runInference(model, processedData);
					console.log(inferenceResults);

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
			img.src = capturedImage;
		} catch (err) {
			console.error('Image processing error:', err);
			setResults({
				success: false,
				error: err.message
			});
			setProcessing(false);
		}
	};

	const toggleRealTimeMode = () => {
		if (!model) {
			alert('Please load a model first');
			return;
		}

		setRealTimeMode(!realTimeMode);
	};

	useEffect(() => {
		const ImageToInference = async () => {
			try {
				const imageSrc = webcamRef.current.getScreenshot();
				setCapturedImage(imageSrc);

				if (imageSrc) {
					const img = new Image();
					img.onload = async () => {
						try {
							let processedData = undefined
							if (size) {
								processedData = preprocessSingleForMobileNet(img, canvasRef.current);
							}
							else {
								processedData = preprocessImage(img, canvasRef.current);
							}
							const inferenceResults = await runInference(model, processedData);

							setRealTimeResults({
								success: true,
								embeddings: inferenceResults,
								timestamp: new Date().toISOString()
							});
						} catch (err) {
							console.error('Real-time processing error:', err);
						}
					};
					img.src = imageSrc;

					setTimeout(() => {
						processCapture(true)
					}, 300);
				}
				else {
					console.error('Failed to capture image for real-time processing');
				}
			} catch (err) {
				console.error('Real-time capture error:', err);
			}
		}

		ImageToInference()

		if (realTimeMode && isWebcamActive && model) {
			intervalRef.current = setInterval(ImageToInference, 5000); // Process every second
		} else if (intervalRef.current) {
			clearInterval(intervalRef.current);
			intervalRef.current = null;
		}

		return () => {
			if (intervalRef.current) {
				clearInterval(intervalRef.current);
			}
		};
	}, [realTimeMode, isWebcamActive, model]);

	return (
		<div className="webcam-capture">
			<h3>Webcam Input</h3>

			<div className="webcam-controls">
				{!isWebcamActive ? (
					<button onClick={startWebcam} className="start-webcam-button">
						Start Webcam
					</button>
				) : (
					<div className="active-controls">
						<button onClick={stopWebcam} className="stop-webcam-button">
							Stop Webcam
						</button>
						<button onClick={capturePhoto} className="capture-button">
							Capture Photo
						</button>
						<button
							onClick={toggleRealTimeMode}
							className={`realtime-button ${realTimeMode ? 'active' : ''}`}
							disabled={!model}
						>
							{realTimeMode ? 'Stop Real-time' : 'Start Real-time'}
						</button>
					</div>
				)}
			</div>

			{isWebcamActive && (
				<div className="webcam-container">
					<Webcam
						audio={false}
						ref={webcamRef}
						screenshotFormat="image/jpeg"
						videoConstraints={videoConstraints}
						className="webcam-feed"
					/>

					{realTimeMode && realTimeResults && (
						<div className="realtime-overlay">
							<div className="realtime-info">
								<p>🔴 Real-time processing active</p>
								<p>Embedding dim: {realTimeResults.embeddings?.length || 'N/A'}</p>
								<p>Last update: {new Date(realTimeResults.timestamp).toLocaleTimeString()}</p>
							</div>
						</div>
					)}
				</div>
			)}

			{capturedImage && (
				<div className="captured-image">
					<h4>Captured Image:</h4>
					<img
						src={capturedImage}
						alt="Captured"
						className="capture-preview"
					/>

					<button
						onClick={() => processCapture()}
						disabled={processing || !model}
						className="process-button"
					>
						{processing ? 'Processing...' : 'Process Capture'}
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

export default WebcamCapture;
