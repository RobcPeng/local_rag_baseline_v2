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
        "Machine learning is a branch of AI",
        "Deep learning is a subset of machine learning",
        "Python is a programming language"
    ]
    scores = rerank_model.rerank(query, passages)
    print("\nReranker Test:")
    for score, passage in zip(scores, passages):
        print(f"Score: {score:.3f} | Passage: {passage}")

if __name__ == "__main__":
    print("Testing Models...")
    # test_llama()
    test_embeddings()
    test_reranker()