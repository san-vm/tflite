import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Webcam from 'react-webcam';
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

const getRealtimeMetric = (output, outputType) => {
	if (outputType === 'embeddings') {
		return {
			label: 'Dims',
			value: output?.length ?? 'N/A',
		};
	}

	if (Array.isArray(output)) {
		return {
			label: 'Rows',
			value: output.length,
		};
	}

	if (output && typeof output === 'object') {
		return {
			label: 'Keys',
			value: Object.keys(output).length,
		};
	}

	return {
		label: 'State',
		value: output ? 'Ready' : 'Idle',
	};
};

const RealTimeOverlay = React.memo(({ isActive, output, outputType, lastUpdate, fps }) => {
	if (!isActive) {
		return null;
	}

	const metric = getRealtimeMetric(output, outputType);

	return (
		<div className="realtime-overlay">
			<div className="realtime-info">
				<div className="status-indicator">
					LIVE <span>{new Date(lastUpdate).toLocaleTimeString()}</span>
				</div>
				<div className="metrics">
					<span>{metric.label}: {metric.value}</span>
					<span>FPS: {fps}</span>
				</div>
			</div>
		</div>
	);
});

const OutputDisplay = React.memo(({ output, outputType, isRealTime = false }) => {
	if (!output) {
		return null;
	}

	return (
		<div className={`embedding-display ${isRealTime ? 'realtime' : 'static'}`}>
			<div className="success-results">
				{outputType === 'embeddings' ? (
					<>
						<p><strong>Embedding dimensions:</strong> {output.length}</p>
						<pre className="embedding-data">
							{JSON.stringify(getPreviewPayload(output), null, 2)}
						</pre>
					</>
				) : (
					<pre className="embedding-data">
						{JSON.stringify(getPreviewPayload(output), null, 2)}
					</pre>
				)}
			</div>
		</div>
	);
});

const WebcamCapture = ({ model, modelName }) => {
	const [isWebcamActive, setIsWebcamActive] = useState(false);
	const [capturedImage, setCapturedImage] = useState(null);
	const [results, setResults] = useState(null);
	const [processing, setProcessing] = useState(false);
	const [realTimeMode, setRealTimeMode] = useState(false);
	const [realTimeResults, setRealTimeResults] = useState(null);
	const [fps, setFps] = useState(0);

	const webcamRef = useRef(null);
	const canvasRef = useRef(null);
	const outputCanvasRef = useRef(null);
	const animationFrameRef = useRef(null);
	const lastProcessTimeRef = useRef(0);
	const frameCountRef = useRef(0);
	const lastFpsUpdateRef = useRef(Date.now());
	const processingRef = useRef(false);

	const modelDetails = getModelConfig(modelName);
	const supportsRealTime = modelDetails?.output.type !== 'image';

	const videoConstraints = useMemo(
		() => ({
			width: 640,
			height: 480,
			facingMode: 'user',
			frameRate: { ideal: 30, max: 60 },
		}),
		[]
	);

	const processImageData = useCallback(
		async (imageSrc, updateResults = true, targetCanvas = null) => {
			if (!model || !modelDetails || processingRef.current) {
				return null;
			}

			processingRef.current = true;

			try {
				return await new Promise((resolve, reject) => {
					const img = new Image();
					img.onload = async () => {
						try {
							const processedData = prepareModelInput(
								img,
								canvasRef.current,
								modelDetails
							);
							const startTime = performance.now();
							const inferenceOutput = await runInference(
								model,
								processedData,
								modelDetails
							);
							const inferenceTimeMs = performance.now() - startTime;

							const result = {
								success: true,
								output: inferenceOutput,
								inferenceTimeMs,
								timestamp: new Date().toISOString(),
							};

							if (modelDetails.output.type === 'image' && targetCanvas) {
								result.outputDimensions = renderImageOutputToCanvas(
									inferenceOutput,
									targetCanvas,
									img.width,
									img.height
								);
							}

							if (updateResults) {
								setRealTimeResults(result);
							}

							resolve(result);
						}
						catch (err) {
							console.error('Processing error:', err);
							const errorResult = { success: false, error: err.message };

							if (updateResults) {
								setRealTimeResults(errorResult);
							}

							reject(errorResult);
						}
						finally {
							processingRef.current = false;
						}
					};
					img.onerror = () => {
						processingRef.current = false;
						reject(new Error('Failed to load image'));
					};
					img.src = imageSrc;
				});
			}
			catch (err) {
				processingRef.current = false;
				throw err;
			}
		},
		[model, modelDetails]
	);

	const processRealTime = useCallback(() => {
		if (!realTimeMode || !isWebcamActive || !webcamRef.current || !supportsRealTime) {
			return;
		}

		const now = Date.now();

		if (now - lastProcessTimeRef.current >= 33) {
			try {
				const imageSrc = webcamRef.current.getScreenshot();
				if (imageSrc && !processingRef.current) {
					processImageData(imageSrc, true).catch(console.error);
					lastProcessTimeRef.current = now;

					frameCountRef.current += 1;
					if (now - lastFpsUpdateRef.current >= 1000) {
						setFps(
							Math.round(
								(frameCountRef.current * 1000) /
								(now - lastFpsUpdateRef.current)
							)
						);
						frameCountRef.current = 0;
						lastFpsUpdateRef.current = now;
					}
				}
			}
			catch (err) {
				console.error('Real-time capture error:', err);
			}
		}

		animationFrameRef.current = requestAnimationFrame(processRealTime);
	}, [isWebcamActive, processImageData, realTimeMode, supportsRealTime]);

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

	const capturePhoto = useCallback(() => {
		if (!webcamRef.current) {
			return;
		}

		const imageSrc = webcamRef.current.getScreenshot();
		setCapturedImage(imageSrc);
		setResults(null);
	}, []);

	const processCapture = useCallback(async () => {
		if (!capturedImage || !model) {
			alert('Please capture an image and load a model first');
			return;
		}

		setProcessing(true);
		setResults(null);

		try {
			const result = await processImageData(
				capturedImage,
				false,
				outputCanvasRef.current
			);
			setResults(result);
		}
		catch (err) {
			setResults({
				success: false,
				error: err.message,
			});
		}
		finally {
			setProcessing(false);
		}
	}, [capturedImage, model, processImageData]);

	const toggleRealTimeMode = useCallback(() => {
		if (!model) {
			alert('Please load a model first');
			return;
		}

		if (!supportsRealTime) {
			return;
		}

		setRealTimeMode((prev) => !prev);
	}, [model, supportsRealTime]);

	const startWebcam = () => {
		setIsWebcamActive(true);
		setCapturedImage(null);
		setResults(null);
	};

	useEffect(() => {
		if (!supportsRealTime) {
			setRealTimeMode(false);
		}
	}, [supportsRealTime]);

	useEffect(() => {
		if (realTimeMode && isWebcamActive && model && supportsRealTime) {
			processRealTime();
		}
		else if (animationFrameRef.current) {
			cancelAnimationFrame(animationFrameRef.current);
			animationFrameRef.current = null;
		}

		return () => {
			if (animationFrameRef.current) {
				cancelAnimationFrame(animationFrameRef.current);
			}
		};
	}, [isWebcamActive, model, processRealTime, realTimeMode, supportsRealTime]);

	useEffect(() => {
		startWebcam();

		return () => {
			if (animationFrameRef.current) {
				cancelAnimationFrame(animationFrameRef.current);
			}
		};
	}, []);

	if (!modelDetails) {
		return null;
	}

	return (
		<div className="webcam-capture">
			<div className="card-header">
				<h2>Live Feed</h2>
			</div>

			<div className="webcam-controls">
				<div className="active-controls">
					<button
						onClick={isWebcamActive ? stopWebcam : startWebcam}
						className="stop-webcam-button"
					>
						{isWebcamActive ? 'Stop Webcam' : 'Start Webcam'}
					</button>
					<button onClick={capturePhoto} className="capture-button">
						Capture Photo
					</button>
					<button
						onClick={toggleRealTimeMode}
						className={`realtime-button ${realTimeMode ? 'active' : ''}`}
						disabled={!model || !supportsRealTime}
					>
						{realTimeMode ? 'Stop Real-time' : 'Start Real-time'}
					</button>
				</div>
				{!supportsRealTime && (
					<p className="webcam-note">
						Real-time mode is disabled for image-output models. Use Capture Photo to inspect the processed image.
					</p>
				)}
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
							isActive={realTimeMode && Boolean(realTimeResults?.success)}
							output={realTimeResults?.output}
							outputType={modelDetails.output.type}
							lastUpdate={realTimeResults?.timestamp}
							fps={fps}
						/>

						{realTimeMode && realTimeResults?.success && (
							<OutputDisplay
								output={realTimeResults.output}
								outputType={modelDetails.output.type}
								isRealTime={true}
							/>
						)}
					</div>
				)}

				{capturedImage && (
					<div className="captured-image">
						<h4>Captured Image</h4>
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

				{modelDetails.output.type === 'image' && (
					<div className="output-image-section">
						<h4>Output Image</h4>
						<canvas ref={outputCanvasRef} className="output-canvas" />
					</div>
				)}

				{results && (
					<div className="results-section">
						<h4>Processing Results</h4>
						{results.success ? (
							<div className="success-results">
								{modelDetails.output.type === 'image' ? (
									<>
										<p>Image processing completed successfully.</p>
										<p><strong>Output resolution:</strong> {results.outputDimensions.width} x {results.outputDimensions.height}</p>
									</>
								) : (
									<OutputDisplay
										output={results.output}
										outputType={modelDetails.output.type}
									/>
								)}
								<p><strong>Inference time:</strong> {(results.inferenceTimeMs / 1000).toFixed(2)} s</p>
								<p><strong>Processed at:</strong> {new Date(results.timestamp).toLocaleString()}</p>
							</div>
						) : (
							<div className="error-results">
								<p>Processing failed: {results.error}</p>
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
