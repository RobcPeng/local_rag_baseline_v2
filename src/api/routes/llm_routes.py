from flask import Blueprint, request, jsonify
from src.utils.llama_init import LlamaModel

bp = Blueprint('llm', __name__, url_prefix='/api/llm')
model = LlamaModel()

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    response = model.chat(messages)
    return jsonify(response)

@bp.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    response = model.generate(prompt)
    return jsonify(response)