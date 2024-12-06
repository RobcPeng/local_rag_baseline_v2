from elasticsearch import Elasticsearch
from typing import Dict, List, Optional, Union, Any
from src.utils.model_utilities import EmbeddingModel
import logging

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, host="http://localhost:9200"):
        self.client = Elasticsearch(host)
        self.logger = logging.getLogger(__name__)
        self.embedding_model = EmbeddingModel() 

    def create_index(self, index_name, mappings=None):
        """Create an index with optional mappings"""
        if mappings is None:
            mappings = {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_model.model.config.hidden_size,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "properties": {
                            "source": {"type": "keyword"},
                            "file_type": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "page_number": {"type": "integer"},
                            "processed_date": {"type": "date"},
                            "doc_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "access_level": {"type": "keyword"}
                        }
                    }
                }
            }
        
        try:
            return self.client.indices.create(
                index=index_name,
                mappings=mappings
            )
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {str(e)}")
            raise

    def index_document(self, index_name, document, doc_id=None):
        """Index a single document"""
        try:
            return self.client.index(
                index=index_name,
                document=document,
                id=doc_id,
                refresh=True  # Make document immediately searchable
            )
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise

    def bulk_index(self, index_name, documents):
        """Bulk index multiple documents"""
        try:
            if not documents:
                logger.warning("No documents provided for bulk indexing.")
                return None

            operations = []
            for doc in documents:
                if not isinstance(doc, dict):
                    logger.error(f"Invalid document format: {doc}")
                    continue

                # Ensure essential fields exist
                if "content" not in doc or "embedding" not in doc or "metadata" not in doc:
                    logger.error(f"Document missing required fields: {doc}")
                    continue

                operations.extend([
                    {"index": {"_index": index_name}},
                    doc
                ])

            if not operations:
                logger.warning("No valid documents to index after validation.")
                return None

            # Execute the bulk operation
            response = self.client.bulk(
                operations=operations,
                refresh=True  # Make documents immediately searchable
            )

            # Check for indexing errors
            if response.get('errors', False):
                logger.error(f"Errors occurred during bulk indexing: {response}")
            else:
                logger.info(f"Successfully indexed {len(documents)} documents into index '{index_name}'.")

            return response

        except Exception as e:
            logger.error(f"Error in bulk indexing documents: {str(e)}")
            raise

    def search(self, index_name, query, size=10, filters=None):
        """
        Search with optional filters
        
        Args:
            index_name (str): Name of the index to search
            query (dict): Main query (e.g., {"match": {"content": "search text"}})
            size (int): Number of results to return
            filters (dict): Optional filters (e.g., {"metadata.user_id": "user123"})
        """
        try:
            search_query = {
                "query": {
                    "bool": {
                        "must": [query]
                    }
                },
                "size": size
            }

            if filters:
                filter_clauses = []
                for field, value in filters.items():
                    filter_clauses.append({"term": {field: value}})
                search_query["query"]["bool"]["filter"] = filter_clauses

            logger.debug(f"Executing search query: {search_query}")
            
            return self.client.search(
                index=index_name,
                body=search_query
            )
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return {
                "hits": {
                    "total": {"value": 0},
                    "hits": []
                }
            }

    def vector_search(self, index_name, vector, size=10, filters=None):
        """Perform vector similarity search"""
        try:
            # Verify index existence
            if not self.client.indices.exists(index=index_name):
                logger.error(f"Index '{index_name}' does not exist.")
                return {"hits": {"total": {"value": 0}, "hits": []}}

            # Get mapping and validate embedding field
            mapping = self.client.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings'].get('properties', {})
            embedding_field = properties.get('embedding')

            if not embedding_field:
                logger.error(f"Index '{index_name}' does not have an 'embedding' field.")
                return {"hits": {"total": {"value": 0}, "hits": []}}

            # Validate vector dimensionality if defined
            expected_dims = embedding_field.get('dims')
            if expected_dims and len(vector) != expected_dims:
                logger.error(f"Vector dimension mismatch: Expected {expected_dims}, got {len(vector)}")
                return {"hits": {"total": {"value": 0}, "hits": []}}

            # Construct query
            base_query = {"match_all": {}}  # Default base query
            if filters:
                base_query = {
                    "bool": {
                        "must": {"match_all": {}},
                        "filter": [{"term": {k: v}} for k, v in filters.items()]
                    }
                }

            search_query = {
                "query": {
                    "script_score": {
                        "query": base_query,
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, \"embedding\") + 1.0",
                            "params": {"query_vector": vector}
                        }
                    }
                },
                "size": size
            }

            logger.debug(f"Executing vector search on index '{index_name}' with query: {search_query}")

            # Execute the query
            response = self.client.search(index=index_name, body=search_query)
            logger.info(f"Vector search executed successfully on index '{index_name}'.")
            return response

        except Exception as e:
            logger.error(f"Error during vector search on index '{index_name}': {str(e)}")
            return {"hits": {"total": {"value": 0}, "hits": []}}

    def hybrid_search(self, index_name, text_query, vector=None, size=10, filters=None, text_boost=0.3, vector_boost=0.7):
        """Hybrid search combining text matching and vector similarity"""
        try:
            bool_should = [
                {
                    "match": {
                        "content": {
                            "query": text_query,
                            "boost": text_boost,
                            "operator": "and"
                        }
                    }
                }
            ]

            if vector is not None:
                bool_should.append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": vector}
                        },
                        "boost": vector_boost
                    }
                })

            search_query = {
                "query": {
                    "bool": {
                        "minimum_should_match": 1,
                        "should": bool_should
                    }
                },
                "size": size
            }

            if filters:
                search_query["query"]["bool"]["filter"] = [
                    {"term": {field: value}} 
                    for field, value in filters.items()
                ]

            logger.debug(f"Executing hybrid search query: {search_query}")
            
            return self.client.search(
                index=index_name,
                body=search_query
            )
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return {
                "hits": {
                    "total": {"value": 0},
                    "hits": []
                }
            }
        
    def verify_document_structure(self, index_name: str) -> Dict:
        """
        Verify index and document structure
        Returns dict with verification results
        """
        results = {
            "index_exists": False,
            "mapping_ok": False,
            "documents_found": False,
            "has_embeddings": False,
            "embedding_dims": None,
            "sample_doc": None,
            "doc_count": 0,
            "errors": []
        }

        try:
            # Check if index exists
            results["index_exists"] = self.client.indices.exists(index=index_name)
            if not results["index_exists"]:
                results["errors"].append(f"Index {index_name} does not exist")
                return results

            # Get mapping
            mapping = self.client.indices.get_mapping(index=index_name)
            results["mapping_ok"] = True
            
            # Get document count
            count = self.client.count(index=index_name)
            results["doc_count"] = count["count"]
            results["documents_found"] = count["count"] > 0

            if results["documents_found"]:
                # Get sample document
                sample = self.client.search(
                    index=index_name,
                    body={
                        "query": {"match_all": {}},
                        "size": 1
                    }
                )
                
                if sample["hits"]["hits"]:
                    sample_doc = sample["hits"]["hits"][0]["_source"]
                    results["sample_doc"] = sample_doc
                    
                    # Check for embeddings
                    if "embedding" in sample_doc:
                        results["has_embeddings"] = True
                        results["embedding_dims"] = len(sample_doc["embedding"])
                    else:
                        results["errors"].append("Documents don't have embedding field")

            # Log verification results
            self.logger.info(f"Index verification results: {results}")
            
            return results

        except Exception as e:
            results["errors"].append(f"Error during verification: {str(e)}")
            self.logger.error(f"Error verifying document structure: {e}")
            return results

    def get_index_stats(self, index_name: str) -> Dict:
        """Get index statistics"""
        try:
            return self.client.indices.stats(index=index_name)
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {}

    def check_document(self, index_name: str, doc_id: str) -> Dict:
        """Retrieve and check specific document"""
        try:
            return self.client.get(index=index_name, id=doc_id)
        except Exception as e:
            self.logger.error(f"Error retrieving document {doc_id}: {e}")
            return {}