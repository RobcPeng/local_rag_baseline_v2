from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(f"Available GPUs: {torch.cuda.device_count()}")
print(torch.version.cuda)

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


class EmbeddingModel:
    def __init__(self):
        self.config = self._load_config("embed_settings.yml")
        # Clear GPU memory before loading model
        torch.cuda.empty_cache()
        gc.collect()
        self.tokenizer, self.model = self._init_model()

    def _load_config(self, config_file):
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "config" / config_file
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _init_model(self):
        model_path = self.config['paths']['model']['dir']
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['download_url'],
            cache_dir=model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )

        # Calculate memory limits for each GPU (using 80% of available memory)
        n_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(n_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = f"{int(total_mem * 0.8 / 1024**2)}MB"
        max_memory['cpu'] = "32GB"  # Allow CPU offloading
        
        print(f"Setting max memory config: {max_memory}")

        # Load model with explicit memory constraints
        model = AutoModel.from_pretrained(
            self.config['model']['download_url'],
            cache_dir=model_path,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=torch.float16,  # Use half precision
            trust_remote_code=self.config['model']['trust_remote_code']
        )

        return tokenizer, model

    def embed(self, texts, **kwargs):
        batch_size = min(self.config['model']['batch_size'], 8)
        chunk_size = 1000
        all_embeddings = []
        
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_texts = texts[chunk_start:chunk_start + chunk_size]
            
            # Tokenize current chunk
            inputs = self.tokenizer(
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            chunk_embeddings = []  # Initialize the list here
            for i in range(0, len(chunk_texts), batch_size):
                torch.cuda.empty_cache()
                gc.collect()
                
                batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
                device = next(self.model.parameters()).device
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                
                with torch.no_grad():
                    try:
                        outputs = self.model(**batch_inputs)
                        print(f"Model output keys: {outputs.keys()}")  # Debug print
                        
                        # Handle model output based on its structure
                        if isinstance(outputs, dict):
                            if 'embeddings' in outputs:
                                emb = outputs['embeddings']
                            elif 'hidden_states' in outputs:
                                emb = outputs['hidden_states'][-1].mean(dim=1)
                            elif 'last_hidden_state' in outputs:
                                emb = outputs['last_hidden_state'].mean(dim=1)
                            else:
                                # Try to get the first tensor that could be embeddings
                                emb = None
                                for key, value in outputs.items():
                                    if isinstance(value, torch.Tensor) and value.dim() >= 2:
                                        emb = value.mean(dim=1)
                                        print(f"Using output key: {key}")  # Debug print
                                        break
                                if emb is None:
                                    raise ValueError(f"Could not find suitable embedding tensor in outputs. Available keys: {outputs.keys()}")
                        else:
                            emb = outputs.mean(dim=1)
                        
                        chunk_embeddings.append(emb.cpu())
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            gc.collect()
                            half_batch = len(batch_inputs['input_ids']) // 2
                            if half_batch == 0:
                                raise
                            print(f"OOM error, retrying with batch size {half_batch}")
                            return self.embed(texts, batch_size=half_batch)
                        else:
                            raise
            
            all_embeddings.append(torch.cat(chunk_embeddings, dim=0))
        
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings

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
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model']['download_url'],
            cache_dir=model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['download_url'],
            cache_dir=model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected. Using DataParallel for RerankModel.")
            model = torch.nn.DataParallel(model)

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