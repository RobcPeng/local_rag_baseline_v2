from flask import Blueprint, request, jsonify
from src.utils.elasticsearch_utils import ElasticsearchClient

bp = Blueprint('search', __name__, url_prefix='/api/search')
es_client = ElasticsearchClient()

@bp.route('/semantic', methods=['POST'])
def semantic_search():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        query = data.get('query', '')
        filters = data.get('filters', {})
        size = data.get('size', 10)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        results = es_client.semantic_search(
            index_name='documents',
            query_text=query,
            filters=filters,
            size=size
        )
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Search route error: {str(e)}")
        return jsonify({
            "hits": {
                "hits": [],
                "total": {"value": 0}
            }
        })