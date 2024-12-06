from flask import Blueprint, request, jsonify
from src.utils.llama_init import LlamaModel
from src.utils.elasticsearch_utils import ElasticsearchClient
from src.utils.model_utilities import RerankModel
from langchain.prompts import PromptTemplate
import torch

bp = Blueprint('rag', __name__, url_prefix='/api/rag')

class RAGService:
    def __init__(self, system_prompt=None):
        self.llm = LlamaModel()
        self.es_client = ElasticsearchClient()
        self.reranker = RerankModel()
        
        self.default_system_prompt = "You are a helpful assistant that answers questions based on the provided context."
        self.system_prompt = system_prompt or self.default_system_prompt
        
        self.default_prompt_template = """Answer the following question based on the provided context. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {question}

Answer: """

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_prompt_template(self, template: str):
        self.prompt_template = PromptTemplate.from_template(template)

    def retrieve(self, query: str, k: int = 5) -> list:
        """Retrieve relevant documents and rerank them"""
        initial_results = self.es_client.search(
            index_name="documents",
            query={"match": {"content": query}},
            size=k*2
        )
        
        passages = [hit['_source']['content'] for hit in initial_results['hits']['hits']]
        
        if len(passages) > 1:
            scores = self.reranker.rerank(query, passages)
            scored_passages = list(zip(scores, passages))
            scored_passages.sort(reverse=True)
            return [p for _, p in scored_passages[:k]]
        
        return passages[:k]

    def generate_response(self, question: str, context: str, prompt_template: str = None) -> str:
        """Generate response using LLM with optional custom prompt"""
        template = prompt_template or self.default_prompt_template
        prompt = PromptTemplate.from_template(template).format(
            context=context,
            question=question
        )
        
        response = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        return response['choices'][0]['message']['content']

@bp.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question']
        k = data.get('k', 5)
        system_prompt = data.get('system_prompt')
        prompt_template = data.get('prompt_template')
        
        rag = RAGService(system_prompt=system_prompt)
        
        # Retrieve and combine contexts
        contexts = rag.retrieve(question, k)
        combined_context = "\n\n".join(contexts)
        
        # Generate response with optional custom prompt
        response = rag.generate_response(
            question, 
            combined_context,
            prompt_template=prompt_template
        )
        
        return jsonify({
            'answer': response,
            'contexts': contexts,
            'system_prompt': rag.system_prompt,
            'prompt_template': prompt_template or rag.default_prompt_template
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@bp.route('/custom_query', methods=['POST'])
def custom_query():
    """Endpoint for fully customizable RAG queries"""
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        # Required parameters
        question = data['question']
        
        # Optional parameters with defaults
        config = {
            'k': data.get('k', 5),
            'system_prompt': data.get('system_prompt', None),
            'prompt_template': data.get('prompt_template', None),
            'temperature': data.get('temperature', 0.7),
            'max_tokens': data.get('max_tokens', None),
            'search_filters': data.get('filters', {}),
            'rerank_results': data.get('rerank', True)
        }
        
        rag = RAGService(system_prompt=config['system_prompt'])
        
        # Retrieve contexts
        contexts = rag.retrieve(question, config['k'])
        combined_context = "\n\n".join(contexts)
        
        # Generate response with all configurations
        response = rag.generate_response(
            question=question,
            context=combined_context,
            prompt_template=config['prompt_template']
        )
        
        return jsonify({
            'answer': response,
            'contexts': contexts,
            'config': {
                'system_prompt': rag.system_prompt,
                'prompt_template': config['prompt_template'] or rag.default_prompt_template,
                'k': config['k'],
                'temperature': config['temperature'],
                'max_tokens': config['max_tokens']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()