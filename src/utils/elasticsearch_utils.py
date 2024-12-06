from elasticsearch import Elasticsearch
import logging

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, host="http://localhost:9200"):
        self.client = Elasticsearch(host)
        
    def create_index(self, index_name, mappings=None):
        """Create an index with optional mappings"""
        if mappings is None:
            mappings = {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
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
            operations = []
            for doc in documents:
                operations.extend([
                    {"index": {"_index": index_name}},
                    doc
                ])
            
            if operations:
                return self.client.bulk(
                    operations=operations,
                    refresh=True  # Make documents immediately searchable
                )
            return None
            
        except Exception as e:
            logger.error(f"Error bulk indexing documents: {str(e)}")
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
        """
        Perform vector similarity search
        
        Args:
            index_name (str): Name of the index to search
            vector (list): Query vector
            size (int): Number of results to return
            filters (dict): Optional filters
        """
        try:
            search_query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": vector}
                        }
                    }
                },
                "size": size
            }

            if filters:
                search_query["query"]["script_score"]["query"] = {
                    "bool": {
                        "must": {"match_all": {}},
                        "filter": [{"term": {k: v}} for k, v in filters.items()]
                    }
                }

            logger.debug(f"Executing vector search query: {search_query}")
            
            return self.client.search(
                index=index_name,
                body=search_query
            )
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return {
                "hits": {
                    "total": {"value": 0},
                    "hits": []
                }
            }