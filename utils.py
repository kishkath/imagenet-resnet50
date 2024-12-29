import os
import logging
import torch
import shutil
import psutil
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

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f'Error creating directory {path}: {str(e)}')
        raise

# Alias for backward compatibility
ensure_dir_exists = ensure_directory

def check_gpu_availability():
    """Check GPU availability and return details"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            return True, gpu_name, memory
        return False, None, 0
    except Exception as e:
        print(f"Error checking GPU availability: {str(e)}")
        return False, None, 0

def check_system_resources():
    """Check if system has sufficient resources"""
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 8 * 1024 * 1024 * 1024:  # 8GB
            print("Warning: Less than 8GB of RAM available")
            return False
            
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 50 * 1024 * 1024 * 1024:  # 50GB
            print("Warning: Less than 50GB of disk space available")
            return False
            
        # Check GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 8:  # 8GB
                print("Warning: GPU has less than 8GB memory")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error checking system resources: {str(e)}")
        return False

def cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 5):
    """Clean up old checkpoints, keeping only the specified number of recent ones"""
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
            
        checkpoints = list(checkpoint_dir.glob('*.ckpt'))
        if len(checkpoints) <= keep_last:
            return
            
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove older checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint.name}")
            
    except Exception as e:
        print(f"Error cleaning up checkpoints: {str(e)}")

def save_backup(filename: str):
    """Create a backup of important files"""
    try:
        if not os.path.exists(filename):
            return
            
        # Create backups directory
        backup_dir = Path('backups')
        backup_dir.mkdir(exist_ok=True)
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"{filename.replace('.', '_')}_{timestamp}.bak"
        
        shutil.copy2(filename, backup_path)
        print(f"Created backup: {backup_path}")
        
    except Exception as e:
        print(f"Error creating backup of {filename}: {str(e)}")

# Alias for backward compatibility
clean_old_checkpoints = cleanup_checkpoints 