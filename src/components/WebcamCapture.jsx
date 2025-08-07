import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import Webcam from 'react-webcam';
import { preprocessImage, preprocessSingleForMobileNet, runInference } from '../utils/modelUtils';

// Separate component for real-time overlay to prevent unnecessary re-renders
const RealTimeOverlay = React.memo(({ isActive, embeddings, lastUpdate, fps }) => {
	if (!isActive) return null;

	return (
		<div className="realtime-overlay">
			<div className="realtime-info">
				<div className="status-indicator">🔴 LIVE
					<span>{'  '}{new Date(lastUpdate).toLocaleTimeString()}</span>
				</div>
				<div className="metrics">
					<span>Emd: {embeddings?.length || 'N/A'}</span>
					<span>{' '}FPS: {fps}</span>
				</div>
			</div>
		</div>
	);
});

// Separate component for embedding display with optimized rendering
const EmbeddingDisplay = React.memo(({ embeddings, timestamp, isRealTime = false }) => {
	const displayData = useMemo(() => {
		if (!embeddings) return null;
		return embeddings.slice(0, 10);
	}, [embeddings]);

	if (!embeddings) return null;

	return (
		<div className={`embedding-display ${isRealTime ? 'realtime' : 'static'}`}>
			<div className="success-results">
				{displayData.map((value, index) => (
					<span key={index} className="embedding-value">
						{typeof value === 'number' ? value.toFixed(4) : value}
					</span>
				))}
				{embeddings.length > 10 && <span className="more-indicator">... +{embeddings.length - 10}</span>}
			</div>
		</div>
	);
});

// Main component optimized for real-time performance
const WebcamCapture = ({ model, size }) => {
	const [isWebcamActive, setIsWebcamActive] = useState(false);
	const [capturedImage, setCapturedImage] = useState(null);
	const [results, setResults] = useState(null);
	const [processing, setProcessing] = useState(false);
	const [realTimeMode, setRealTimeMode] = useState(false);
	const [realTimeResults, setRealTimeResults] = useState(null);
	const [fps, setFps] = useState(0);

	const webcamRef = useRef(null);
	const canvasRef = useRef(null);
	const animationFrameRef = useRef(null);
	const lastProcessTimeRef = useRef(0);
	const frameCountRef = useRef(0);
	const lastFpsUpdateRef = useRef(Date.now());
	const processingRef = useRef(false);

	const videoConstraints = useMemo(() => ({
		width: 640,
		height: 480,
		facingMode: "user",
		frameRate: { ideal: 30, max: 60 }
	}), []);

	// Optimized image processing function
	const processImageData = useCallback(async (imageSrc, updateResults = true) => {
		if (!model || processingRef.current) return null;

		processingRef.current = true;

		try {
			return new Promise((resolve, reject) => {
				const img = new Image();
				img.onload = async () => {
					try {
						const processedData = size
							? preprocessSingleForMobileNet(img, canvasRef.current)
							: preprocessImage(img, canvasRef.current);

						const inferenceResults = await runInference(model, processedData);

						const result = {
							success: true,
							embeddings: inferenceResults,
							timestamp: Date.now()
						};

						if (updateResults) {
							setRealTimeResults(result);
						}

						resolve(result);
					} catch (err) {
						console.error('Processing error:', err);
						const errorResult = { success: false, error: err.message };
						if (updateResults) {
							setRealTimeResults(errorResult);
						}
						reject(errorResult);
					} finally {
						processingRef.current = false;
					}
				};
				img.onerror = () => {
					processingRef.current = false;
					reject(new Error('Failed to load image'));
				};
				img.src = imageSrc;
			});
		} catch (err) {
			processingRef.current = false;
			throw err;
		}
	}, [model, size]);

	// Real-time processing loop using requestAnimationFrame
	const processRealTime = useCallback(() => {
		if (!realTimeMode || !isWebcamActive || !webcamRef.current) {
			return;
		}

		const now = Date.now();

		// Throttle to ~30 FPS for processing while maintaining smooth UI
		if (now - lastProcessTimeRef.current >= 33) { // ~30 FPS
			try {
				const imageSrc = webcamRef.current.getScreenshot();
				if (imageSrc && !processingRef.current) {
					processImageData(imageSrc, true).catch(console.error);
					lastProcessTimeRef.current = now;

					// Update FPS counter
					frameCountRef.current++;
					if (now - lastFpsUpdateRef.current >= 1000) {
						setFps(Math.round(frameCountRef.current * 1000 / (now - lastFpsUpdateRef.current)));
						frameCountRef.current = 0;
						lastFpsUpdateRef.current = now;
					}
				}
			} catch (err) {
				console.error('Real-time capture error:', err);
			}
		}

		animationFrameRef.current = requestAnimationFrame(processRealTime);
	}, [realTimeMode, isWebcamActive, processImageData]);

	const stopWebcam = useCallback(() => {
		setIsWebcamActive(false);
		setRealTimeMode(false);
		setRealTimeResults(null);
		if (animationFrameRef.current) {
			cancelAnimationFrame(animationFrameRef.current);
			animationFrameRef.current = null;
		}
		processingRef.current = false;
	}, []);

	// Capture photo
	const capturePhoto = useCallback(() => {
		if (!webcamRef.current) return;
		const imageSrc = webcamRef.current.getScreenshot();
		setCapturedImage(imageSrc);
		setResults(null);
	}, []);

	// Process captured image
	const processCapture = useCallback(async () => {
		if (!capturedImage || !model) {
			alert('Please capture an image and load a model first');
			return;
		}

		setProcessing(true);
		setResults(null);

		try {
			const result = await processImageData(capturedImage, false);
			setResults(result);
		} catch (err) {
			setResults({
				success: false,
				error: err.message
			});
		} finally {
			setProcessing(false);
		}
	}, [capturedImage, model, processImageData]);

	// Toggle real-time mode
	const toggleRealTimeMode = useCallback(() => {
		if (!model) {
			alert('Please load a model first');
			return;
		}
		setRealTimeMode(prev => !prev);
	}, [model]);

	// Effect to handle real-time processing
	useEffect(() => {
		if (realTimeMode && isWebcamActive && model) {
			processRealTime();
		} else if (animationFrameRef.current) {
			cancelAnimationFrame(animationFrameRef.current);
			animationFrameRef.current = null;
		}

		return () => {
			if (animationFrameRef.current) {
				cancelAnimationFrame(animationFrameRef.current);
			}
		};
	}, [realTimeMode, isWebcamActive, model, processRealTime]);

	const startWebCameFunc = () => {
		setIsWebcamActive(true);
		setCapturedImage(null);
		setResults(null);
	}

	// Cleanup on unmount
	useEffect(() => {
		startWebCameFunc();

		return () => {
			if (animationFrameRef.current) {
				cancelAnimationFrame(animationFrameRef.current);
			}
		};
	}, []);

	return (
		<div className="webcam-capture" style={{ display: "flex", width: "800px" }}>
			<div className="webcam-controls" style={{ width: "100px" }}>
				<div className="active-controls">
					<button onClick={isWebcamActive ? stopWebcam : startWebCameFunc} className="stop-webcam-button">
						{isWebcamActive ? "Stop Webcam" : "Start Webcam"}
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
			</div>
			<div>

				{isWebcamActive && (
					<div className="webcam-container">
						<Webcam
							audio={false}
							ref={webcamRef}
							screenshotFormat="image/jpeg"
							videoConstraints={videoConstraints}
							className="webcam-feed"
						/>

						<RealTimeOverlay
							isActive={realTimeMode}
							embeddings={realTimeResults?.embeddings}
							lastUpdate={realTimeResults?.timestamp}
							fps={fps}
						/>

						{/* Real-time embeddings display */}
						{realTimeMode && realTimeResults?.success && (
							<EmbeddingDisplay
								embeddings={realTimeResults.embeddings}
								timestamp={realTimeResults.timestamp}
								isRealTime={true}
							/>
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
							onClick={processCapture}
							disabled={processing || !model}
							className="process-button"
						>
							{processing ? 'Processing...' : 'Process Capture'}
						</button>
					</div>
				)}

				{/* Static results for captured image */}
				{results && (
					<div className="results-section">
						<h4>Processing Results:</h4>
						{results.success ? (
							<div className="success-results">
								<p>✅ Face processing completed successfully!</p>
								<EmbeddingDisplay
									embeddings={results.embeddings}
									timestamp={results.timestamp}
									isRealTime={false}
								/>
							</div>
						) : (
							<div className="error-results">
								<p>❌ Processing failed: {results.error}</p>
							</div>
						)}
					</div>
				)}
			</div>

			<canvas ref={canvasRef} style={{ display: 'none' }} />

		</div>
	);
};

export default WebcamCapture;
