from flask import Blueprint, request, jsonify
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangchainDocument
from src.utils.elasticsearch_utils import ElasticsearchClient
from src.utils.model_utilities import EmbeddingModel, RerankModel
from pydantic import Field, BaseModel
from typing import List, Dict, Any, Tuple
import logging
import torch

logger = logging.getLogger(__name__)

bp = Blueprint('ollama', __name__, url_prefix='/peng')

class StaticRetriever(BaseRetriever, BaseModel):
    documents: List[LangchainDocument] = Field(default_factory=list)
    
    def get_relevant_documents(self, _) -> List[LangchainDocument]:
        return self.documents
    

    async def aget_relevant_documents(self, _) -> List[LangchainDocument]:
        return self.documents

def get_ollama_llm(temperature: float = 0):
    """Initialize Ollama with correct base URL"""
    return OllamaLLM(
        model="mistral",  # or configure from settings
        temperature=temperature,
        base_url="http://localhost:11434"
    )

def setup_rag_chain(retriever, system_prompt: str = None, temperature: float = None) -> RetrievalQA:
    prompt_template = f"""{system_prompt or "You are a helpful assistant answering questions based on the provided context."}

Documents: {{context}}

Question: {{question}}

Instructions: Return only plain text - do not use line breaks or escape characters. Use the Context as your only knowledge base. Do not make up answers if you do not know.

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = get_ollama_llm(temperature if temperature is not None else 0)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT
        },
        return_source_documents=True
    )

def rerank_chunks(query: str, chunks: List[str], k: int = 3) -> List[Dict]:
    if not chunks:
        return []
        
    # Clean and normalize chunks while maintaining original format
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and len(chunk.strip()) > 20]
    
    if not cleaned_chunks:
        return []
    
    rerank_model = RerankModel()
    
    try:
        scores = rerank_model.rerank(query, cleaned_chunks)
        
        # Keep exact same output format
        ranked_chunks = [
            {
                "chunk": chunk,
                "score": float(score),
                "length": len(chunk)
            }
            for chunk, score in zip(cleaned_chunks, scores)
        ]
        
        return sorted(ranked_chunks, key=lambda x: x['score'], reverse=True)[:k]
        
    except Exception as e:
        # Fallback with same format
        return [
            {
                "chunk": chunk,
                "score": 0.5,
                "length": len(chunk)
            }
            for chunk in cleaned_chunks[:k]
        ]

@bp.route('/rag', methods=['POST'])
def perform_rag_query() -> Tuple[Dict[str, Any], int]:
    """Process a RAG query request."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question']
        system_prompt = data.get('system_prompt')
        n_results = data.get('n_results', 5)
        temperature = data.get('temperature', 0.0)
        
        # Initialize Elasticsearch client
        es_client = ElasticsearchClient()
        es_client.verify_document_structure("documents")
        embedding_model = EmbeddingModel()
        
        # Get initial results from Elasticsearch
        initial_results = es_client.search(
            index_name="documents",
            query={"match": {"content": question}},
            size=n_results * 2
        )

        query_embedding = embedding_model.embed([question])[0].tolist()
        logger.error(f"embedding -  {query_embedding}, got {embedding_model.embed([question])}")

        vector_results = es_client.vector_search(
            index_name="documents",
            vector=query_embedding,
            size=n_results * 2
        )

        seen_docs = set()
        combined_hits = []
        for hit in initial_results.get('hits', {}).get('hits', []):
            doc_id = hit['_source'].get('metadata', {}).get('doc_id')
            if doc_id not in seen_docs:
                combined_hits.append(hit)
                seen_docs.add(doc_id)
        
        # Add vector results if available
        if vector_results and vector_results.get('hits', {}).get('hits'):
            for hit in vector_results['hits']['hits']:
                doc_id = hit['_source'].get('metadata', {}).get('doc_id')
                if doc_id not in seen_docs:
                    combined_hits.append(hit)
                    seen_docs.add(doc_id)

        chunks_to_rerank = [hit['_source'].get('content', "") for hit in combined_hits]
        metadata = [hit['_source'].get('metadata', {}) for hit in combined_hits]
        
        # Rerank if we have multiple chunks
        if len(chunks_to_rerank) > 1:
            reranked_results = rerank_chunks(question, chunks_to_rerank, k=n_results)
            docs = [chunk['chunk'] for chunk in reranked_results]
            scores = [chunk['score'] for chunk in reranked_results]
            metadata = metadata[:n_results]
        else:
            docs = chunks_to_rerank
            scores = [1.0] * len(chunks_to_rerank)
        
        # Convert to LangChain documents
        langchain_docs = [
            LangchainDocument(
                page_content=doc,
                metadata=meta
            )
            for doc, meta in zip(docs, metadata)
        ]
        
        # Setup and run RAG chain
        retriever = StaticRetriever(documents=langchain_docs)
        chain = setup_rag_chain(retriever, system_prompt, temperature)
        rag_result = chain.invoke({"query": question})
        
        # Prepare response
        response = {
            'question': question,
            'answer': rag_result['result'],
            'source_documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in zip(langchain_docs, scores)
            ]        
        }
            
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in rag_query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/chat', methods=['POST'])
def direct_chat():
    """Direct chat with Ollama (no RAG)"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        temperature = data.get('temperature', 0.7)
        llm = get_ollama_llm(temperature)
        response = llm.invoke(data['message'])
        
        return jsonify({
            'response': response,
            'model': 'mistral'
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500