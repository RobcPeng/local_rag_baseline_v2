from flask import Flask
from flask_cors import CORS
from src.api.routes.llm_routes import bp as llm_bp
from src.api.routes.embed_routes import bp as embed_bp
from src.api.routes.document_routes import bp as document_bp
from src.api.routes.search_routes import bp as search_bp
from src.api.routes.rerank_routes import bp as rerank_bp
from src.api.routes.rag_routes import bp as rag_bp

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(llm_bp)
    app.register_blueprint(embed_bp)
    app.register_blueprint(document_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(rerank_bp)  
    app.register_blueprint(rag_bp)  


    return app