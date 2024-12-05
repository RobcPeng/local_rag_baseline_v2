from src.utils.llama_init import LlamaModel
from src.utils.model_utilities import EmbeddingModel, RerankModel
import torch

def test_llama():
    llm = LlamaModel()
    response = llm.chat([{"role": "user", "content": "Write a short poem about AI"}])
    print("\nLLM Test:")
    print(response['choices'][0]['message']['content'])

def test_embeddings():
    embed_model = EmbeddingModel()
    texts = ["This is a test sentence", "Another test sentence"]
    embeddings = embed_model.embed(texts)
    print("\nEmbedding Test:")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding device: {torch.tensor(embeddings).device}")

def test_reranker():
    rerank_model = RerankModel()
    query = "What is machine learning?"
    passages = [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Python is a popular programming language used in data science.",
        "AI and machine learning are transforming many industries today.",
        "Data science combines statistics, programming, and domain expertise."
    ]
    
    scored_passages = rerank_model.rerank(query, passages)
    print("\nReranker Test:")
    print(f"Query: {query}\n")
    for score, passage in zip(scored_passages, passages):
        print(f"Score: {score:.3f} | {passage}")

if __name__ == "__main__":
    print("Testing Models...")
    # test_llama()
    test_embeddings()
    test_reranker()