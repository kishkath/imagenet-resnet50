import os
import shutil
import torch
import psutil
import logging
from pathlib import Path
from datetime import datetime

class TrainingError(Exception):
    """Custom exception for training-related errors"""
    pass

class DatasetError(Exception):
    """Custom exception for dataset-related errors"""
    pass

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {str(e)}")
        raise

def check_gpu_availability():
    """Check GPU availability and return details"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, gpu_name, memory
        return False, None, 0
    except Exception as e:
        print(f"Error checking GPU availability: {str(e)}")
        return False, None, 0

def safe_cuda_memory_check():
    """Safely check CUDA memory usage"""
    try:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_cached = torch.cuda.memory_reserved(0) / 1e9
            return memory_allocated, memory_cached
        return 0, 0
    except Exception as e:
        print(f"Error checking CUDA memory: {str(e)}")
        return 0, 0

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

def cleanup_checkpoints(checkpoint_dir, keep_last=5):
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

def save_backup(filename):
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

def setup_logging(log_file='training.log'):
    """Setup logging configuration"""
    try:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise 