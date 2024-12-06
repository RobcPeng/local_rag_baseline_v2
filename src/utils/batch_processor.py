from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
from .document_processor import DocumentProcessor
from .elasticsearch_utils import ElasticsearchClient

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.es_client = ElasticsearchClient()
        self.batch_size = 50

    def process_directory(
        self,
        directory: str | Path,
        custom_metadata: Optional[Dict] = None,
        recursive: bool = True,
        file_types: List[str] = None,
        index_name: str = "documents"
    ):
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
            
        if file_types is None:
            file_types = ['.pdf', '.txt', '.docx', '.doc', '.md']
            
        # Get all files
        pattern = '**/*' if recursive else '*'
        files = []
        for file_type in file_types:
            files.extend(directory.glob(f"{pattern}{file_type}"))
        
        if not files:
            logger.warning(f"No matching files found in {directory}")
            return
        
        # Ensure index exists
        if not self.es_client.client.indices.exists(index=index_name):
            logger.info(f"Creating index: {index_name}")
            self.es_client.create_index(index_name)
        
        # Process files
        total_documents = 0
        failed_files = []
        all_documents = []  # Store all documents before batching
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                logger.info(f"Processing {file_path}")
                
                # Create file-specific metadata
                file_metadata = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix,
                    "document_id": str(hash(str(file_path))),  # Add unique document identifier
                    **(custom_metadata or {})
                }
                
                # Process single file
                documents = self.doc_processor.process_file(file_path, file_metadata)
                
                if documents:
                    # Update metadata for each chunk to maintain document identity
                    for doc in documents:
                        doc['metadata'].update(file_metadata)
                    all_documents.extend(documents)
                    
                else:
                    logger.warning(f"No documents extracted from {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                failed_files.append((str(file_path), str(e)))
                continue
        
        # Batch index all documents
        for i in range(0, len(all_documents), self.batch_size):
            batch = all_documents[i:i + self.batch_size]
            self.es_client.bulk_index(index_name, batch)
            total_documents += len(batch)
            
        summary = {
            "total_files": len(files),
            "processed_files": len(files) - len(failed_files),
            "failed_files": len(failed_files),
            "total_documents_indexed": total_documents,
            "failures": failed_files
        }
        
        logger.info("Processing complete:")
        logger.info(f"Total files: {summary['total_files']}")
        logger.info(f"Successfully processed: {summary['processed_files']}")
        logger.info(f"Failed: {summary['failed_files']}")
        logger.info(f"Total documents indexed: {summary['total_documents_indexed']}")
        
        if failed_files:
            logger.info("Failed files:")
            for file_path, error in failed_files:
                logger.info(f"- {file_path}: {error}")
                
        return summary
    
    def refresh_index(self, index_name: str = "documents"):
        """Force refresh of the index to make documents searchable immediately"""
        self.es_client.client.indices.refresh(index=index_name)