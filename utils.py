import os
import logging
import torch
from pathlib import Path
from datetime import datetime

class DatasetError(Exception):
    pass

class TrainingError(Exception):
    pass

def setup_logging(name='training', log_dir='logs'):
    """Setup logging configuration"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for unique log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        # Configure logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler with simple formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.info(f'Logging setup complete. Logs will be saved to: {log_file}')
        return logger
        
    except Exception as e:
        print(f'Error setting up logging: {str(e)}')
        raise

def safe_cuda_memory_check():
    """Safely check CUDA memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            cached = torch.cuda.memory_reserved() / 1024**3
            return allocated, cached
        return 0, 0
    except Exception:
        return 0, 0

def ensure_dir_exists(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f'Error creating directory {path}: {str(e)}')
        raise

def clean_old_checkpoints(checkpoint_dir: str, keep_n: int = 3) -> None:
    """Clean old checkpoints, keeping only the n most recent ones"""
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
            
        checkpoints = sorted(
            checkpoint_dir.glob('*.ckpt'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[keep_n:]:
            checkpoint.unlink()
            
    except Exception as e:
        print(f'Error cleaning checkpoints: {str(e)}') 