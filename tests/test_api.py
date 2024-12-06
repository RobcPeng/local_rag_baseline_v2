import pytest
from src.api import create_app
import json
import os
from pathlib import Path
import torch
import gc

@pytest.fixture(autouse=True)
def clear_gpu_memory():
    """Automatically clear GPU memory after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.mark.gpu
def test_llm_chat(client, clear_gpu_memory):
    try:
        response = client.post('/api/llm/chat', 
            json={
                'messages': [{'role': 'user', 'content': 'Hello!'}]
            }
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'choices' in data
        assert len(data['choices']) > 0
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@pytest.mark.gpu
def test_embedding(client, clear_gpu_memory):
    try:
        response = client.post('/api/embed/encode',
            json={
                'texts': ['This is a test sentence']
            }
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'embeddings' in data
        assert 'dimension' in data
        assert len(data['embeddings']) > 0
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@pytest.mark.gpu
def test_rerank(client, clear_gpu_memory):
    try:
        response = client.post('/api/rerank/score',
            json={
                'query': 'What is machine learning?',
                'passages': [
                    "Machine learning is a branch of artificial intelligence.",
                    "Python is a programming language.",
                    "Deep learning is a subset of machine learning.",
                ]
            }
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        assert len(data['results']) == 3
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def test_document_processing(client, test_dirs):
    # Use the test directories fixture
    test_file_path = test_dirs['data'] / "test.txt"
    
    try:
        # Create test file
        test_file_path.write_text("This is a test document.")
        
        # Verify file exists
        assert test_file_path.exists(), f"Test file not created at {test_file_path}"
        
        with open(test_file_path, 'rb') as f:
            response = client.post('/api/documents/process',
                data={
                    'file': (f, 'test.txt'),
                    'metadata': json.dumps({'user_id': 'test_user'})
                }
            )
        
        # Print response details if there's an error
        if response.status_code != 200:
            print(f"Response Status: {response.status_code}")
            print(f"Response Data: {response.data.decode()}")
            
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'message' in data
        
    except Exception as e:
        print(f"Test error: {str(e)}")
        raise
        
    finally:
        # Cleanup
        if test_file_path.exists():
            test_file_path.unlink()

@pytest.mark.gpu
def test_semantic_search(client):
    response = client.post('/api/search/semantic',
        json={
            'query': 'test query',
            'filters': {'user_id': 'test_user'},
            'size': 10
        }
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'hits' in data
    assert 'hits' in data['hits']
    assert isinstance(data['hits']['hits'], list)

@pytest.mark.gpu
def test_rag_query(client, clear_gpu_memory):
    try:
        response = client.post('/api/rag/query',
            json={
                'question': 'What is machine learning?',
                'k': 3
            }
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'answer' in data
        assert 'contexts' in data
        assert isinstance(data['contexts'], list)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@pytest.mark.gpu
def test_rag_custom_query(client, clear_gpu_memory):
    try:
        response = client.post('/api/rag/custom_query',
            json={
                'question': 'What is machine learning?',
                'system_prompt': 'You are an AI expert.',
                'prompt_template': 'Context:\n{context}\n\nQ: {question}\nA:',
                'k': 3,
                'temperature': 0.8
            }
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'answer' in data
        assert 'contexts' in data
        assert 'config' in data
        assert data['config']['system_prompt'] == 'You are an AI expert.'
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()