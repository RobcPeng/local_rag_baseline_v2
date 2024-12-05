from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from pathlib import Path
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmbeddingModel:
    def __init__(self):
        self.config = self._load_config("embed_settings.yml")
        self.model = self._init_model()

    def _load_config(self, config_file):
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "config" / config_file
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _init_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.config['paths']['model']['dir']
        return SentenceTransformer(
            self.config['model']['download_url'],
            device=device,
            trust_remote_code= self.config['model']['trust_remote_code'],
            cache_folder=model_path

        )

    def embed(self, texts, **kwargs):
        return self.model.encode(
            texts,
            batch_size=self.config['model']['batch_size'],
            **kwargs
        )

class RerankModel:
    def __init__(self):
        self.config = self._load_config("rerank_settings.yml")
        self.model = self._init_model()
        

    def _load_config(self, config_file):
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "config" / config_file
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _init_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.config['paths']['model']['dir']
        
        AutoModelForSequenceClassification.from_pretrained(
            self.config['model']['download_url'],
            cache_dir=model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )

        AutoTokenizer.from_pretrained(
            self.config['model']['download_url'],
            cache_dir=model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        return CrossEncoder(
            model_name=self.config['model']['download_url'],
            device=device,
            max_length=self.config['model']['max_length']
        )

    def rerank(self, query, passages):
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(
            pairs,
            batch_size=self.config['model']['batch_size']
        )
        print(scores)
        return scores