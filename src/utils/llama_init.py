from llama_cpp import Llama
import torch
from pathlib import Path
import requests
from tqdm import tqdm
from .config_loader import load_config
import gc
import logging

logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self):
        self.config = load_config()
        self._ensure_model_exists()
        self.clear_gpu_memory()
        self.llm = self._init_model()

    def clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _download_model(self):
        url = self.config.get('model', {}).get('download_url')
        if not url:
            raise ValueError("Model download URL not configured")
        
        self.config['paths']['model_path'].mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.config['paths']['model_file'], 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

    def _ensure_model_exists(self):
        if not self.config['paths']['model_file'].exists():
            logger.info(f"Model not found at {self.config['paths']['model_file']}")
            self._download_model()

    def _init_model(self) -> Llama:
        try:
            base_config = {
                "model_path": str(self.config['paths']['model_file']),
                "n_ctx": self.config['model']['n_ctx'],
                "verbose": True,
                "n_batch": min(32, self.config['model']['cuda'].get('n_batch', 32)),
                "n_gpu_layers": min(24, self.config['model'].get('n_gpu_layers', 24)),
                "main_gpu": 0,
                "offload_kqv": True,  # Offload key/query/value matrices to CPU when not in use
                "use_mmap": True,  # Use memory mapping for faster loading
                "use_mlock": False  # Don't lock memory in RAM
            }

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                logger.info(f"Initializing model with {torch.cuda.device_count()} GPUs")
                memory_per_gpu = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"GPU memory available: {memory_per_gpu / 1024**3:.2f} GB")
                
                return Llama(
                    **base_config,
                    tensor_split=[0.5, 0.5]  # Split between GPUs
                )
            
            elif torch.cuda.is_available():
                logger.info("Initializing model with single GPU")
                memory_available = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"GPU memory available: {memory_available / 1024**3:.2f} GB")
                
                return Llama(**base_config)
            
            else:
                logger.info("No GPU available, using CPU")
                return Llama(
                    model_path=str(self.config['paths']['model_file']),
                    n_ctx=self.config['model']['n_ctx'],
                    verbose=True
                )
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def generate(self, prompt: str, **kwargs):
        try:
            self.clear_gpu_memory()
            return self.llm(
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                **kwargs
            )
        finally:
            self.clear_gpu_memory()

    def chat(self, messages: list, **kwargs):
        try:
            self.clear_gpu_memory()
            return self.llm.create_chat_completion(
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 512),
                **kwargs
            )
        finally:
            self.clear_gpu_memory()