import React, { useState } from 'react';
import { MODEL_OPTIONS } from '../config/modelRegistry';

const FaceDetection = () => {
	const [savedEmbeddings, setSavedEmbeddings] = useState([]);
	const [comparisonResults, setComparisonResults] = useState(null);
	const embeddingModels = MODEL_OPTIONS.filter(
		(model) => model.output.type === 'embeddings'
	);

	const clearSavedEmbeddings = () => {
		setSavedEmbeddings([]);
		setComparisonResults(null);
	};

	return (
		<div className="face-detection">
			<div className="card-header">
				<h2>Embedding Vault</h2>
			</div>

			<div className="saved-embeddings">
				<h4>Saved Face Embeddings: {savedEmbeddings.length}</h4>

				{savedEmbeddings.length > 0 && (
					<div className="embeddings-list">
						{savedEmbeddings.map(item => (
							<div key={item.id} className="embedding-item">
								<p><strong>{item.label}</strong></p>
								<p>Saved: {new Date(item.timestamp).toLocaleString()}</p>
								<p>Dimensions: {item.embedding.length}</p>
							</div>
						))}

						<button
							onClick={clearSavedEmbeddings}
							className="clear-embeddings-button"
						>
							Clear All
						</button>
					</div>
				)}
			</div>

			{comparisonResults && (
				<div className="comparison-results">
					<h4>Face Recognition Results:</h4>
					{comparisonResults.map((result, index) => (
						<div
							key={index}
							className={`comparison-item ${result.similarity > 0.8 ? 'high-match' :
								result.similarity > 0.6 ? 'medium-match' : 'low-match'}`}
						>
							<p><strong>{result.label}</strong></p>
							<p>Similarity: {(result.similarity * 100).toFixed(2)}%</p>
							<p>
								Match Level: {
									result.similarity > 0.8 ? '🟢 High' :
										result.similarity > 0.6 ? '🟡 Medium' : '🔴 Low'
								}
							</p>
						</div>
					))}
				</div>
			)}

			<div className="detection-info">
				<h4>How to use</h4>
				<ol>
					<li>Load an embedding model such as MobileFaceNet or FaceNet-512.</li>
					<li>Process face images or webcam captures to get embedding vectors.</li>
					<li>Save embeddings with labels for recognition workflows.</li>
					<li>Compare new faces against saved embeddings.</li>
				</ol>

				<div className="model-comparison">
					<h5>Embedding Models</h5>
					<ul>
						{embeddingModels.map((model) => (
							<li key={model.id}>
								<strong>{model.name} ({model.size}):</strong> {model.description}
							</li>
						))}
					</ul>
				</div>

				<p>
					Image-output models and generic raw-output models are available in the playground, but they are not connected to this embedding comparison panel.
				</p>
			</div>
		</div>
	);
};

export default FaceDetection;
