from flask import Blueprint, request, jsonify
from src.utils.document_processor import DocumentProcessor
from pathlib import Path
import logging
import json
import traceback
from src.utils.elasticsearch_utils import ElasticsearchClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

bp = Blueprint('documents', __name__, url_prefix='/api/documents')
processor = DocumentProcessor()

@bp.route('/process', methods=['POST'])
def process_document():
    try:
        if 'file' not in request.files:
            logger.error("No file provided")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        try:
            custom_metadata = json.loads(request.form.get('metadata', '{}'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid metadata JSON: {e}")
            return jsonify({'error': 'Invalid metadata format'}), 400
        
        # Get absolute path
        current_dir = Path(__file__).parent.parent.parent.parent
        temp_dir = current_dir / "data" / "documents"
        temp_path = temp_dir / file.filename
        
        logger.debug(f"Processing file: {temp_path}")
        
        # Ensure directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file.save(str(temp_path))
        logger.debug(f"File saved successfully to {temp_path}")
        
        try:
            # Process document
            logger.debug("Starting document processing")
            documents = processor.process_file(temp_path, custom_metadata)
            logger.debug(f"Document processed into {len(documents)} chunks")
            
            # Index documents
            if documents:
                logger.debug("Starting document indexing")
                processor.index_documents(documents, 'documents')
                logger.debug("Indexing completed")
            
            return jsonify({
                'message': 'Document processed successfully',
                'num_chunks': len(documents) if documents else 0
            })
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Document processing error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup
        try:
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path}")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

@bp.route('/status', methods=['GET'])
def check_status():
    es_client = ElasticsearchClient()
    verification = es_client.verify_document_structure("documents")
    return jsonify(verification)