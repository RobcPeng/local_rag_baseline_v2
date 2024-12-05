import yaml
from pathlib import Path

def load_config():
    root_dir = Path(__file__).parent.parent.parent
    config_path = root_dir / "config" / "settings.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config['paths']['project_root'] = root_dir
    config['paths']['model_path'] = root_dir / config['paths']['model']['dir']
    config['paths']['model_file'] = config['paths']['model_path'] / config['paths']['model']['file']
    
    return config