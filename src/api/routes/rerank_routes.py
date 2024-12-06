from flask import Blueprint, request, jsonify
from src.utils.model_utilities import RerankModel

bp = Blueprint('rerank', __name__, url_prefix='/api/rerank')
model = RerankModel()

@bp.route('/score', methods=['POST'])
def rerank():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    query = data.get('query')
    passages = data.get('passages', [])
    
    if not query or not passages:
        return jsonify({'error': 'Query and passages are required'}), 400
    
    try:
        scores = model.rerank(query, passages)
        # Create a list of scored passages
        scored_results = [
            {
                'score': float(score),  # Convert numpy float to Python float
                'passage': passage
            }
            for score, passage in zip(scores, passages)
        ]
        
        # Sort by score in descending order
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'results': scored_results,
            'original_order': list(range(len(passages)))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500