import pytest
from pathlib import Path
import shutil

@pytest.fixture(scope="session")
def test_dirs():
    """Create and manage test directories"""
    # Get project root directory
    root_dir = Path(__file__).parent.parent
    
    # Define directories
    dirs = {
        'data': root_dir / 'data' / 'documents',
        'processed': root_dir / 'data' / 'processed',
        'test_data': root_dir / 'tests' / 'data',
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    yield dirs
    
    # Cleanup after tests
    for dir_path in dirs.values():
        if dir_path.exists():
            for file in dir_path.iterdir():
                if file.is_file():
                    file.unlink()