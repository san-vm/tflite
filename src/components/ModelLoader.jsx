import React from 'react';

const ModelLoader = ({
	models,
	selectedModel,
	setSelectedModel,
	loadModel,
	isLoading,
	error
}) => {
	const modelInfo = {
		mobileFaceNet: {
			name: 'MobileFaceNet',
			size: '10MB',
			description: 'Lightweight face recognition model optimized for mobile devices'
		},
		faceNet512: {
			name: 'FaceNet-512',
			size: '24MB',
			description: 'High-accuracy face recognition with 512-dimensional embeddings'
		},
		mirnet_int8: {
			name: 'mirnet_int8',
			size: '24MB',
			description: 'High-accuracy face recognition with 512-dimensional embeddings'
		},
	};

	return (
		<div className="model-loader">
			<h3>Model Management</h3>

			<div className="model-selection">
				<label>Select Model:</label>
				<select
					value={selectedModel}
					onChange={(e) => setSelectedModel(e.target.value)}
					disabled={isLoading}
				>
					<option value="mobileFaceNet">MobileFaceNet (10MB)</option>
					<option value="faceNet512">FaceNet-512 (24MB)</option>
					<option value="mirnet_int8">mirnet_int8 (24MB)</option>
				</select>
			</div>

			<div className="model-info">
				<h4>{modelInfo[selectedModel].name}</h4>
				<p><strong>Size:</strong> {modelInfo[selectedModel].size}</p>
				<p>{modelInfo[selectedModel].description}</p>
			</div>

			<div className="model-status">
				<p><strong>Status:</strong>
					{models[selectedModel] ?
						<span className="loaded"> Loaded ✅</span> :
						<span className="not-loaded"> Not Loaded ❌</span>
					}
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

			{error && (
				<div className="error-message">
					<p>⚠️ {error}</p>
				</div>
			)}
		</div>
	);
};

export default ModelLoader;
