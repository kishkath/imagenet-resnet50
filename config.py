from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
    """Configuration for ImageNet training"""
    
    # Data paths
    data_dir: str = os.path.join(os.getcwd(), 'data/imagenet')
    save_dir: str = os.path.join(os.getcwd(), 'checkpoints')
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 100
    num_workers: int = 4
    pin_memory: bool = True
    distributed: bool = False
    
    # Model parameters
    num_classes: int = 1000
    gradient_clip_val: float = 1.0
    
    # Optimizer parameters
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # OneCycleLR parameters
    max_lr: float = 0.1
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    
    # Image parameters
    image_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    # Augmentation parameters
    use_autoaugment: bool = True
    use_cutmix: bool = True
    use_mixup: bool = True
    cutmix_alpha: float = 1.0
    mixup_alpha: float = 0.2
    
    # Mixed precision settings
    use_amp: bool = True
    
    # Distributed training settings
    world_size: int = 1
    rank: int = 0
    
    # Logging settings
    show_progress_bar: bool = True
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    save_freq: int = 1
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization"""
        try:
            # Convert paths to Path objects
            self.data_dir = Path(self.data_dir)
            self.save_dir = Path(self.save_dir)
            
            # Validate paths
            if not self.data_dir.exists():
                print(f"Warning: Data directory {self.data_dir} does not exist")
                
            # Create save directory if it doesn't exist
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            # Adjust batch size based on available GPU memory
            if self.distributed:
                print("Distributed training enabled")
                self.batch_size = self.batch_size // self.world_size
                print(f"Adjusted batch size per GPU: {self.batch_size}")
            
            # Validate learning rates
            if self.max_lr < self.learning_rate:
                print("Warning: max_lr should be greater than or equal to learning_rate")
                self.max_lr = self.learning_rate
            
            # Validate worker count
            if self.num_workers > os.cpu_count():
                print(f"Warning: Reducing num_workers from {self.num_workers} to {os.cpu_count()}")
                self.num_workers = os.cpu_count()
                
        except Exception as e:
            print(f"Error in config post-initialization: {str(e)}")
            raise
            
    def print_config(self):
        """Print the current configuration"""
        try:
            print("\nCurrent Configuration:")
            print("="*50)
            for key, value in self.__dict__.items():
                print(f"{key}: {value}")
            print("="*50)
        except Exception as e:
            print(f"Error printing configuration: {str(e)}")
            
    @classmethod
    def from_dict(cls, config_dict):
        """Create a Config instance from a dictionary"""
        try:
            return cls(**config_dict)
        except Exception as e:
            print(f"Error creating config from dictionary: {str(e)}")
            raise 