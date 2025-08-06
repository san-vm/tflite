import React, { useState } from 'react';
import { calculateSimilarity, normalizeEmbedding } from '../utils/modelUtils';

const FaceDetection = () => {
	const [savedEmbeddings, setSavedEmbeddings] = useState([]);
	const [comparisonResults, setComparisonResults] = useState(null);

	const saveEmbedding = (embedding, label) => {
		if (!embedding || !label) return;

		const normalizedEmbedding = normalizeEmbedding(embedding);
		const newEmbedding = {
			id: Date.now(),
			label: label,
			embedding: normalizedEmbedding,
			timestamp: new Date().toISOString()
		};

		setSavedEmbeddings(prev => [...prev, newEmbedding]);
	};

	const compareEmbeddings = (newEmbedding) => {
		if (!newEmbedding || savedEmbeddings.length === 0) return;

		const normalizedNew = normalizeEmbedding(newEmbedding);

		const similarities = savedEmbeddings.map(saved => {
			const similarity = calculateSimilarity(normalizedNew, saved.embedding);
			return {
				label: saved.label,
				similarity: similarity,
				timestamp: saved.timestamp
			};
		});

		// Sort by similarity (highest first)
		similarities.sort((a, b) => b.similarity - a.similarity);

		setComparisonResults(similarities);
	};

	const clearSavedEmbeddings = () => {
		setSavedEmbeddings([]);
		setComparisonResults(null);
	};

	return (
		<div className="face-detection">
			<h3>Face Recognition Database</h3>

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
				<h4>How to use:</h4>
				<ol>
					<li>Load either MobileFaceNet or FaceNet-512 model</li>
					<li>Process face images or use webcam to get embeddings</li>
					<li>Save embeddings with labels for face recognition</li>
					<li>Compare new faces against saved embeddings</li>
				</ol>

				<div className="model-comparison">
					<h5>Model Comparison:</h5>
					<ul>
						<li><strong>MobileFaceNet (10MB):</strong> Faster, optimized for mobile, good accuracy</li>
						<li><strong>FaceNet-512 (24MB):</strong> Higher accuracy, 512D embeddings, more computational</li>
					</ul>
				</div>
			</div>
		</div>
	);
};

export default FaceDetection;
