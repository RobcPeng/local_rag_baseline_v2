from flask import Blueprint, request, jsonify
from src.utils.model_utilities import EmbeddingModel
import numpy as np

bp = Blueprint('embed', __name__, url_prefix='/api/embed')
model = EmbeddingModel()

@bp.route('/encode', methods=['POST'])
def encode():
    data = request.json
    texts = data.get('texts', [])
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    try:
        embeddings = model.embed(texts)
        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = embeddings.numpy().tolist() if hasattr(embeddings, 'numpy') else embeddings.tolist()
        return jsonify({
            'embeddings': embeddings_list,
            'dimension': len(embeddings_list[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500