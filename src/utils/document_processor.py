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
import numpy as np


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = EmbeddingModel()
        self.es_client = Elasticsearch("http://localhost:9200")
        
    def process_file(self, file_path: Union[str, Path], custom_metadata: Dict = None) -> List[Dict]:
        """
        Process a single file and return chunks with metadata
        """
        file_path = Path(file_path)
        file_type = magic.from_file(str(file_path), mime=True)

        # Extract text based on file type
        if file_type == "application/pdf":
            if self._is_scanned_pdf(file_path):
                text = self._extract_with_ocr(file_path)
            else:
                text = self._extract_pdf_text(file_path)
        elif file_type.startswith("image/"):
            text = self._extract_with_ocr(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = self._extract_docx_text(file_path)
        else:
            elements = partition(filename=str(file_path))
            text = "\n".join([str(element) for element in elements])
        
        # Prepare documents with metadata
        documents = []
        chunks = self._create_chunks(text)
        
        for chunk_index, chunk in enumerate(chunks):
            doc_id = self._generate_doc_id(file_path, chunk, chunk_index)
            
            # Get embedding and handle potential NaN values
            embedding = self.embedding_model.embed([chunk])[0]
            # Convert to Python float and replace NaN with 0
            embedding = [float(x) if not np.isnan(x) else 0.0 for x in embedding.tolist()]
            
            metadata = {
                "source": str(file_path),
                "file_type": file_type,
                "chunk_index": chunk_index,
                "processed_date": datetime.now().isoformat(),
                "doc_id": doc_id
            }
            
            if custom_metadata:
                metadata.update(custom_metadata)
            
            documents.append({
                "content": chunk,
                "embedding": embedding,
                "metadata": metadata
            })
        
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

    def _extract_with_ocr(self, file_path: Path) -> str:
        """Extract text from images and scanned PDFs using OCR"""
        text_content = []
        
        # If PDF, convert to images first
        if file_path.suffix.lower() == '.pdf':
            # Convert PDF to images
            pdf_images = pdf2image.convert_from_path(file_path)
            
            for image in pdf_images:
                # Preprocess image
                processed_image = self._preprocess_image(image)
                # Perform OCR
                text_content.append(pytesseract.image_to_string(processed_image))
                
        # For image files
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            image = Image.open(file_path)
            processed_image = self._preprocess_image(image)
            text_content.append(pytesseract.image_to_string(processed_image))
            
        return "\n".join(text_content)
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
        return "\n".join(text_content)
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks, ensuring words are not broken.
        Returns non-empty chunks that maintain word boundaries.
        """
        chunks = []
        if not text.strip():
            return chunks
            
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate initial end point
            end = min(start + self.chunk_size, text_length)
            
            # If we're not at the text end, adjust to not break words
            if end < text_length:
                # Find the last space or newline before the end
                while end > start and not text[end - 1].isspace():
                    end -= 1
                # If no space found in chunk, find the next space
                if end <= start:
                    end = min(start + self.chunk_size, text_length)
                    # If we're still not at the end, find the next space
                    if end < text_length:
                        next_space = text.find(' ', end)
                        if next_space != -1:
                            end = next_space
            
            # Extract the chunk and clean it
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            if end >= text_length:
                break
                
            # Move back from end by overlap amount
            start = max(0, end - self.chunk_overlap)
            # Ensure we start at a word boundary
            while start < text_length and start > 0 and not text[start - 1].isspace():
                start += 1

        return chunks
        
    def _generate_doc_id(self, file_path: Path, chunk: str, chunk_index: int) -> str:
        """Generate a unique document ID"""
        content = f"{file_path}{chunk}{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def index_documents(self, documents: List[Dict], index_name: str = "documents"):
        """Index documents with their embeddings"""
        # Create index if doesn't exist
        print("local test")
        print(self.embedding_model.model.config.hidden_size)
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
                                "processed_date": {"type": "date"},
                                "doc_id": {"type": "keyword"},
                                "user_id": {"type": "keyword"},
                                "access_level": {"type": "keyword"},
                                "organization": {"type": "keyword"},
                                "classification": {"type": "keyword"}
                            }
                        }
                    }
                }
            )

        # Process in batches
        batch_size = 1
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            operations = []
            
            for doc in batch:
                operations.extend([
                    {"index": {"_index": index_name, "_id": doc["metadata"]["doc_id"]}},
                    {
                        "content": doc["content"],
                        "embedding": doc["embedding"],
                        "metadata": doc["metadata"]
                    }
                ])
            
            # Index batch
            self.es_client.bulk(operations=operations, refresh=True)