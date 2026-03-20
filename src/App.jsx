import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tflite from '@tensorflow/tfjs-tflite/dist/tf-tflite.fesm.js';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import ModelLoader from './components/ModelLoader';
import ImageInput from './components/ImageInput';
import WebcamCapture from './components/WebcamCapture';
import FaceDetection from './components/FaceDetection';
import {
	DEFAULT_MODEL_ID,
	createInitialModelState,
	getModelConfig,
} from './config/modelRegistry';
import './App.css';

function App() {
	const [models, setModels] = useState(createInitialModelState);
	const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL_ID);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState(null);
	const [inputMode, setInputMode] = useState('image'); // 'image' or 'webcam'

	const ref = useRef();
	const currentModel = models[selectedModel];
	const currentModelConfig = getModelConfig(selectedModel);
	const statusTone = error ? 'bad' : currentModel ? 'good' : undefined;
	const statusLabel = error
		? 'Runtime attention required'
		: isLoading
			? `Loading ${currentModelConfig?.name ?? 'model'}...`
			: currentModel
				? `${currentModelConfig?.name ?? 'Model'} loaded locally`
				: 'WASM ready. Load a model to begin.';

	useEffect(() => {
		initializeTensorFlow();
	}, []);

	const initializeTensorFlow = async () => {
		if (ref.current) {
			return;
		}
		ref.current = true;
		try {
			// Set WASM paths for TensorFlow.js backend
			setWasmPaths('/wasm/');
			tflite.setWasmPath('/wasm/');

			// Initialize TensorFlow.js with WASM backend for better performance
			await tf.setBackend('wasm');
			await tf.ready();

			console.log('TensorFlow.js initialized with WASM backend');
		} catch (err) {
			console.error('Failed to initialize TensorFlow.js:', err);
			setError('Failed to initialize TensorFlow.js');
		}
	};

	const loadModel = async (modelName) => {
		setIsLoading(true);
		setError(null);

		try {
			const modelConfig = getModelConfig(modelName);
			if (!modelConfig) {
				throw new Error(`Unknown model: ${modelName}`);
			}

			const modelPath = `/models/${modelConfig.fileName}`;
			console.log(`Loading model from: ${modelPath}`);

			const model = await tflite.loadTFLiteModel(modelPath);
			console.log(`${modelName} loaded successfully`);

			setModels((prev) => ({
				...prev,
				[modelName]: model,
			}));
		}
		catch (err) {
			console.error(`Failed to load ${modelName}:`, err);
			setError(`Failed to load ${modelName}: ${err.message}`);
		}
		finally {
			setIsLoading(false);
		}
	};

	return (
		<>
			<div className="bg-shape" />
			<div className="bg-grid" />
			<div className="App">
				<header className="hero compact">
					<div>
						<p className="tag">Vision Runtime</p>
						<h1>GateFace Model Lab</h1>
						<p className="subtitle">
							Load local TFLite models, validate preprocessing contracts, and inspect image or webcam inference entirely in the browser.
						</p>
					</div>
					<div className="status-pill" data-tone={statusTone}>
						{statusLabel}
					</div>
				</header>

				<main className="container">
					<div className="control-stack">
						<div className="input-mode-selector">
							<div className="card-header">
								<h2>Input Mode</h2>
								<p>Choose whether to test the active model against a still image or a live webcam stream.</p>
							</div>
							<div className="mode-buttons">
								<button
									className={inputMode === 'image' ? 'active' : ''}
									onClick={() => setInputMode('image')}
								>
									Image Upload
								</button>
								<button
									className={inputMode === 'webcam' ? 'active' : ''}
									onClick={() => setInputMode('webcam')}
								>
									Webcam Feed
								</button>
							</div>
						</div>

						<ModelLoader
							models={models}
							selectedModel={selectedModel}
							setSelectedModel={setSelectedModel}
							loadModel={loadModel}
							isLoading={isLoading}
							error={error}
						/>
					</div>

					{inputMode === 'image' ? (
						<ImageInput model={currentModel} modelName={selectedModel} />
					) : (
						<WebcamCapture model={currentModel} modelName={selectedModel} />
					)}

					<FaceDetection />
				</main>

				<footer className="footer">
					<span>React UI + TensorFlow Lite WASM runtime.</span>
					<span className="mono">
						{currentModelConfig ? `Selected profile: ${currentModelConfig.name}` : 'Select a model profile.'}
					</span>
				</footer>
			</div>
		</>
	);
}

export default App;
