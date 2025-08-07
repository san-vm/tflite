import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tflite from '@tensorflow/tfjs-tflite';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import ModelLoader from './components/ModelLoader';
import ImageInput from './components/ImageInput';
import WebcamCapture from './components/WebcamCapture';
import FaceDetection from './components/FaceDetection';
import './App.css';

function App() {
	const [models, setModels] = useState({
		mobileFaceNet: null,
		faceNet512: null
	});
	const [selectedModel, setSelectedModel] = useState('mobileFaceNet');
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState(null);
	const [inputMode, setInputMode] = useState('image'); // 'image' or 'webcam'

	const ref = useRef()

	useEffect(() => {
		initializeTensorFlow();
	}, []);

	const initializeTensorFlow = async () => {
		if (ref.current) {
			return
		}
		ref.current = true
		try {
			// Set WASM paths for TensorFlow.js backend
			setWasmPaths("/wasm/");
			tflite.setWasmPath("/wasm/");

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
			// MobileFaceNet, facenet_512
			let tempModelName = ''
			if (modelName === 'mobileFaceNet') {
				tempModelName = 'MobileFaceNet'
			} else if (modelName === 'faceNet512') {
				tempModelName = 'facenet_512'
			}
			else {
				setError(`Failed to load`);
				setIsLoading(false);
				return
			}
			const modelPath = `/models/${tempModelName}.tflite`;
			console.log(`Loading model from: ${modelPath}`);

			const model = await tflite.loadTFLiteModel(modelPath);
			console.log(`${modelName} loaded successfully`);

			setModels(prev => ({
				...prev,
				[modelName]: model
			}));
		} catch (err) {
			console.error(`Failed to load ${modelName}:`, err);
			setError(`Failed to load ${modelName}: ${err.message}`);
		} finally {
			setIsLoading(false);
		}
	};

	const getCurrentModel = () => {
		return models[selectedModel];
	};

	return (
		<div className="App">
			<header className="App-header">
				<h1>TensorFlow Lite Face Detection</h1>
				<p>Local Face Recognition with MobileFaceNet & FaceNet-512</p>
			</header>

			<div className="container">
				{/* Model Selection and Loading */}
				<div>
					{/* Input Mode Selection */}
					<div className="input-mode-selector">
						<h3>Input Mode</h3>
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
								Webcam
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


				{/* Input Components */}
				{inputMode === 'image' ? (
					<ImageInput model={getCurrentModel()} size={selectedModel === "mobileFaceNet"} />
				) : (
					<WebcamCapture model={getCurrentModel()} size={selectedModel === "mobileFaceNet"} />
				)}
			</div>

			{/* Face Detection Results */}
			<FaceDetection />
		</div>
	);
}

export default App;
