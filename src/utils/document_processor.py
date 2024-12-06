from pathlib import Path
import unstructured
from unstructured.partition.auto import partition
from typing import List, Dict, Union, Optional
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch
from src.utils.model_utilities import EmbeddingModel
import torch
import docx
import pypdf
import magic
from tqdm import tqdm
import pytesseract
from PIL import Image, ImageEnhance
import pdf2image
import tempfile

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = EmbeddingModel()
        self.es_client = Elasticsearch("http://localhost:9200")
        
    def process_file(self, file_path: Union[str, Path], custom_metadata: Dict = None) -> List[Dict]:
        """
        Process a single file and return chunks with metadata
        
        Args:
            file_path: Path to the document
            custom_metadata: Dictionary of custom metadata fields (e.g., {"user_id": "123", "access_level": "confidential"})
        """
        file_path = Path(file_path)
        file_type = magic.from_file(str(file_path), mime=True)
        
        # Extract text based on file type
        if file_type == "application/pdf":
            if self._is_scanned_pdf(file_path):
                pages = self._extract_with_ocr(file_path)
            else:
                pages = self._extract_pdf_with_pages(file_path)
        elif file_type.startswith("image/"):
            pages = self._extract_with_ocr(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            pages = self._extract_docx_with_pages(file_path)
        else:
            # Use unstructured for other file types
            elements = partition(filename=str(file_path))
            pages = [{"text": "\n".join([str(element) for element in elements]), "page": 1}]
        
        # Prepare documents with metadata
        documents = []
        chunk_index = 0
        
        for page in pages:
            chunks = self._create_chunks(page["text"])
            
            for chunk in chunks:
                doc_id = self._generate_doc_id(file_path, chunk, chunk_index)
                
                # Combine default and custom metadata
                metadata = {
                    "source": str(file_path),
                    "file_type": file_type,
                    "chunk_index": chunk_index,
                    "page_number": page["page"],
                    "processed_date": datetime.now().isoformat(),
                    "doc_id": doc_id
                }
                
                # Add custom metadata if provided
                if custom_metadata:
                    metadata.update(custom_metadata)
                
                documents.append({
                    "content": chunk,
                    "metadata": metadata
                })
                chunk_index += 1
        
        return documents

    def _is_scanned_pdf(self, file_path: Path) -> bool:
        """Check if PDF is scanned (image-based) and needs OCR"""
        try:
            # Convert first page to image
            images = pdf2image.convert_from_path(file_path, first_page=1, last_page=1)
            if not images:
                return False
                
            # Try to get text directly from PDF
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                first_page_text = pdf_reader.pages[0].extract_text()
                
            # If PDF has very little text but contains an image, it's likely scanned
            return len(first_page_text.strip()) < 100
            
        except Exception as e:
            print(f"Error checking PDF type: {e}")
            return False

    def _preprocess_image(self, image: Image) -> Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        image = image.convert('L')
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        return image

    def _extract_with_ocr(self, file_path: Path) -> List[Dict]:
        """Extract text from images and scanned PDFs using OCR"""
        pages = []
        
        # If PDF, convert to images first
        if file_path.suffix.lower() == '.pdf':
            # Convert PDF to images
            pdf_images = pdf2image.convert_from_path(file_path)
            
            for page_num, image in enumerate(pdf_images, 1):
                # Preprocess image
                processed_image = self._preprocess_image(image)
                # Perform OCR
                text = pytesseract.image_to_string(processed_image)
                pages.append({
                    "text": text,
                    "page": page_num
                })
                
        # For image files
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            image = Image.open(file_path)
            processed_image = self._preprocess_image(image)
            text = pytesseract.image_to_string(processed_image)
            pages.append({
                "text": text,
                "page": 1
            })
            
        return pages
    
    def _extract_pdf_with_pages(self, file_path: Path) -> List[Dict]:
        """Extract text from PDF files with page numbers"""
        pages = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                pages.append({
                    "text": page.extract_text(),
                    "page": page_num
                })
        return pages
    
    def _extract_docx_with_pages(self, file_path: Path) -> List[Dict]:
        """Extract text from DOCX files with page numbers"""
        doc = docx.Document(file_path)
        current_text = ""
        pages = []
        estimated_page = 1
        chars_per_page = 3000  # Approximate characters per page
        
        for paragraph in doc.paragraphs:
            current_text += paragraph.text + "\n"
            
            # Estimate page breaks based on character count
            if len(current_text) >= chars_per_page:
                pages.append({
                    "text": current_text,
                    "page": estimated_page
                })
                current_text = ""
                estimated_page += 1
        
        # Add remaining text
        if current_text:
            pages.append({
                "text": current_text,
                "page": estimated_page
            })
        
        return pages
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
        return chunks
    
    def _generate_doc_id(self, file_path: Path, chunk: str, chunk_index: int) -> str:
        """Generate a unique document ID"""
        content = f"{file_path}{chunk}{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def index_documents(self, documents: List[Dict], index_name: str):
        """Index documents with their embeddings to Elasticsearch"""
        # Create index if it doesn't exist
        if not self.es_client.indices.exists(index=index_name):
            self.es_client.indices.create(
                index=index_name,
                mappings={
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
                                # Dynamic mapping for custom metadata
                                "user_id": {"type": "keyword"},
                                "access_level": {"type": "keyword"},
                                "organization": {"type": "keyword"},
                                "classification": {"type": "keyword"}
                            }
                        }
                    }
                }
            )
        
        # Process documents in batches
        batch_size = 8
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            
            # Get embeddings for the batch
            texts = [doc["content"] for doc in batch]
            embeddings = self.embedding_model.embed(texts)
            
            # Prepare bulk indexing operations
            operations = []
            for doc, embedding in zip(batch, embeddings):
                operations.extend([
                    {"index": {"_index": index_name, "_id": doc["metadata"]["doc_id"]}},
                    {
                        "content": doc["content"],
                        "embedding": embedding.tolist(),
                        "metadata": doc["metadata"]
                    }
                ])
            
            # Index the batch
            self.es_client.bulk(operations=operations)