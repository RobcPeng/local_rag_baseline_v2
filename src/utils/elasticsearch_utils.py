from elasticsearch import Elasticsearch
from pathlib import Path
import yaml

class ElasticClient:
    def __init__(self):
        self.config = self._load_config()
        self.client = Elasticsearch("http://localhost:9200")
    
    def index_document(self, index_name, document, doc_id=None):
        return self.client.index(
            index=index_name,
            document=document,
            id=doc_id
        )
    
    def search(self, index_name, query, size=10):
        return self.client.search(
            index=index_name,
            query=query,
            size=size
        )

    def create_index(self, index_name, mappings):
        return self.client.indices.create(
            index=index_name,
            mappings=mappings
        )