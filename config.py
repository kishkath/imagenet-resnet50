from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    # Data paths
    data_dir: str = '/path/to/imagenet'  # Update this with your ImageNet path
    train_dir: str = 'train'
    val_dir: str = 'val'
    
    # Training parameters
    batch_size: int = 256
    num_workers: int = 8
    epochs: int = 100
    
    # Model parameters
    model_name: str = 'resnet50'
    num_classes: int = 1000
    
    # Optimizer parameters
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # OneCycleLR parameters
    max_lr: float = 0.1
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    
    # Image parameters
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation parameters
    train_aug_scale: Tuple[float, float] = (0.08, 1.0)
    train_aug_ratio: Tuple[float, float] = (0.75, 1.33)
    
    # Mixed precision
    use_amp: bool = True
    
    # Distributed training
    distributed: bool = True
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    project_name: str = 'imagenet_training'
    
    # Checkpointing
    save_dir: str = 'checkpoints'
    save_freq: int = 1 