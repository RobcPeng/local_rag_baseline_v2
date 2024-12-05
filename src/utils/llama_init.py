from llama_cpp import Llama
import torch
from pathlib import Path
import requests
from tqdm import tqdm
from .config_loader import load_config

class LlamaModel:
    def __init__(self):
        self.config = load_config()
        self._ensure_model_exists()
        self.llm = self._init_model()

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
            print(f"Model not found at {self.config['paths']['model_file']}")
            self._download_model()

    def _init_model(self) -> Llama:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Initializing model with {torch.cuda.device_count()} GPUs")
            return Llama(
                model_path=str(self.config['paths']['model_file']),
                n_ctx=self.config['model']['n_ctx'],
                n_gpu_layers=32,  # Load all layers into GPU memory
                n_batch=self.config['model']['cuda']['n_batch'],
                tensor_split=[0.5, 0.5],  # Split between two GPUs
                verbose=True
            )
        elif torch.cuda.is_available():
            print("Initializing model with single GPU")
            return Llama(
                model_path=str(self.config['paths']['model_file']),
                n_ctx=self.config['model']['n_ctx'],
                n_gpu_layers=32,  # Load all layers into GPU memory
                n_batch=self.config['model']['cuda']['n_batch'],
                verbose=True
            )
        else:
            print("No GPU available, using CPU")
            return Llama(
                model_path=str(self.config['paths']['model_file']),
                n_ctx=self.config['model']['n_ctx'],
                verbose=True
            )

    def generate(self, prompt: str, **kwargs):
        return self.llm(prompt, **kwargs)

    def chat(self, messages: list, **kwargs):
        return self.llm.create_chat_completion(messages=messages, **kwargs)