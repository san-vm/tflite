import React from 'react';
import { MODEL_OPTIONS, getModelConfig } from '../config/modelRegistry';

const ModelLoader = ({
	models,
	selectedModel,
	setSelectedModel,
	loadModel,
	isLoading,
	error,
}) => {
	const modelInfo = getModelConfig(selectedModel);

	if (!modelInfo) {
		return null;
	}

	return (
		<div className="model-loader">
			<div className="card-header">
				<h2>Model Vault</h2>
				<p>Pick a registered model profile, inspect its contract, and load it into the in-browser runtime.</p>
			</div>

			<div className="model-selection">
				<label htmlFor="model-select">Select Model:</label>
				<select
					id="model-select"
					value={selectedModel}
					onChange={(event) => setSelectedModel(event.target.value)}
					disabled={isLoading}
				>
					{MODEL_OPTIONS.map((modelOption) => (
						<option key={modelOption.id} value={modelOption.id}>
							{modelOption.name} ({modelOption.size})
						</option>
					))}
				</select>
			</div>

				<div className="model-status">
				<p><strong>Status:</strong>
					{models[selectedModel] ? (
						<span className="loaded"> Loaded</span>
					) : (
						<span className="not-loaded"> Not Loaded</span>
					)}
				</p>

				<button
					onClick={() => loadModel(selectedModel)}
					disabled={isLoading || models[selectedModel]}
					className="load-button"
				>
					{isLoading ? 'Loading...' :
						models[selectedModel] ? 'Model Loaded' : 'Load Model'}
				</button>
			</div>

			<div className="model-info">
				<h4>{modelInfo.name}</h4>
				<p><strong>Size:</strong> {modelInfo.size}</p>
				<p><strong>Model file:</strong> {`public/models/${modelInfo.fileName}`}</p>
				<p><strong>Input shape:</strong> {modelInfo.input.shape.join(' x ')}</p>
				<p><strong>Output mode:</strong> {modelInfo.output.type}</p>
				<p>{modelInfo.description}</p>
			</div>

			<div className="model-usage">
				<h4>Usage Notes</h4>
				<ul>
					{modelInfo.documentation.map((note) => (
						<li key={note}>{note}</li>
					))}
				</ul>
			</div>

			{error && (
				<div className="error-message">
					<p>{error}</p>
				</div>
			)}
		</div>
	);
};

export default ModelLoader;
