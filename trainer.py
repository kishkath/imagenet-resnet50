import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist
import torch
import logging
from tqdm import tqdm
import sys
from datetime import datetime

# Import local modules
from config import Config
from dataset import ImageNetDataset
from model import ResNet50Module

# Configure logging to display to both console and file
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Add timestamp and make it visually distinct
        record.message = record.getMessage()
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        if record.levelno == logging.INFO:
            return f"\033[92m{timestamp} | {record.message}\033[0m"  # Green color for info
        elif record.levelno == logging.WARNING:
            return f"\033[93m{timestamp} | WARNING: {record.message}\033[0m"  # Yellow for warnings
        elif record.levelno == logging.ERROR:
            return f"\033[91m{timestamp} | ERROR: {record.message}\033[0m"  # Red for errors
        return f"{timestamp} | {record.message}"

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with colors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

# File handler with standard formatting
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class DetailedLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_batch_size = None
        self.val_batch_size = None
        self.current_epoch = 0
        self.training_start_time = None
        
    def on_train_start(self, trainer, pl_module):
        self.training_start_time = datetime.now()
        print("\n" + "="*100)
        print(f"Training started at {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch
        epoch_str = f"Starting Epoch {self.current_epoch+1}/{trainer.max_epochs}"
        print("\n" + "="*100)
        print(f"{epoch_str:^100}")
        print("="*100 + "\n")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:  # Log every 10 batches
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            
            # Calculate elapsed time and estimate remaining time
            elapsed_time = datetime.now() - self.training_start_time
            progress = (self.current_epoch * len(trainer.train_dataloader) + batch_idx) / \
                      (trainer.max_epochs * len(trainer.train_dataloader))
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
            else:
                remaining_time = None
            
            status = (
                f"Epoch: {self.current_epoch+1}/{trainer.max_epochs} | "
                f"Batch: {batch_idx}/{len(trainer.train_dataloader)} | "
                f"Loss: {loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            if remaining_time:
                status += f" | Remaining: {str(remaining_time).split('.')[0]}"
            
            print(status)
            sys.stdout.flush()
            
    def on_validation_epoch_start(self, trainer, pl_module):
        val_str = f"Validation - Epoch {self.current_epoch+1}/{trainer.max_epochs}"
        print("\n" + "-"*100)
        print(f"{val_str:^100}")
        print("-"*100 + "\n")
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 5 == 0:  # Log every 5 validation batches
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            print(f"Validation Batch: {batch_idx}/{len(trainer.val_dataloaders[0])} | Loss: {loss:.4f}")
            sys.stdout.flush()
            
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', 0)
        train_acc = metrics.get('train_acc', 0)
        
        result_str = (
            f"\nEpoch {self.current_epoch+1}/{trainer.max_epochs} Results:\n"
            f"{'='*50}\n"
            f"Training Loss: {train_loss:.4f}\n"
            f"Training Accuracy: {train_acc:.4f}\n"
            f"{'='*50}\n"
        )
        print(result_str)
        sys.stdout.flush()

class ImageNetTrainer:
    def __init__(self, config: Config):
        self.config = config
        logger.info("Initializing ImageNet Trainer...")
        
    def setup(self):
        logger.info("Setting up training...")
        # Initialize dataset
        dataset = ImageNetDataset(self.config)
        self.train_loader, self.val_loader = dataset.get_dataloaders(
            distributed=self.config.distributed
        )
        logger.info(f"Dataset loaded - Training samples: {len(self.train_loader.dataset)}, "
                   f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Initialize model
        self.model = ResNet50Module(config=self.config)
        # Enable gradient checkpointing in the model
        if hasattr(self.model.model, 'set_grad_checkpointing'):
            self.model.model.set_grad_checkpointing(enable=True)
            logger.info("Gradient checkpointing enabled via set_grad_checkpointing")
        else:
            # Alternative way to enable gradient checkpointing for ResNet
            for module in self.model.model.modules():
                if isinstance(module, torch.nn.modules.container.Sequential):
                    module.gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled via module setting")
        
        # Setup logging
        if self.config.use_wandb:
            self.logger = WandbLogger(
                project=self.config.project_name,
                log_model=True,
                group=f"resnet50_{self.config.batch_size}_{self.config.learning_rate}"
            )
            logger.info("WandB logging enabled")
        else:
            self.logger = True
        
        # Setup callbacks
        self.callbacks = [
            DetailedLoggingCallback(),
            ModelCheckpoint(
                dirpath=self.config.save_dir,
                filename='resnet50-{epoch:02d}-{val_acc:.3f}',
                monitor='val_acc',
                mode='max',
                save_top_k=3,
                every_n_epochs=self.config.save_freq
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        logger.info("Callbacks configured")
        
    def train(self):
        logger.info("Starting training...")
        
        # Setup strategy for single GPU training on Kaggle
        strategy = "auto"
        
        # Initialize trainer with optimized settings for Kaggle
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu',
            devices=1,  # Single GPU for Kaggle
            strategy=strategy,
            precision=16 if self.config.use_amp else 32,  # Simplified precision setting
            callbacks=self.callbacks,
            logger=self.logger,
            log_every_n_steps=1,  # Log every step for detailed monitoring
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            deterministic=False,  # Faster training
            benchmark=True,  # Optimize CUDA kernels
            enable_progress_bar=True,
            enable_model_summary=True,
            detect_anomaly=False,  # Faster training
            val_check_interval=self.config.val_check_interval
        )
        
        logger.info(
            f"Training Configuration:\n"
            f"Batch Size: {self.config.batch_size}\n"
            f"Learning Rate: {self.config.learning_rate}\n"
            f"Mixed Precision: {self.config.use_amp}\n"
            f"Gradient Clipping: {self.config.gradient_clip_val}\n"
            f"Validation Check Interval: {self.config.val_check_interval}"
        )
        print(f"Training Configuration:\n"
            f"Batch Size: {self.config.batch_size}\n"
            f"Learning Rate: {self.config.learning_rate}\n"
            f"Mixed Precision: {self.config.use_amp}\n"
            f"Gradient Clipping: {self.config.gradient_clip_val}\n"
            f"Validation Check Interval: {self.config.val_check_interval}"
        )
        
        # Train
        print('fitting trainer...')
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
def train_imagenet(config: Config):
    print("In train_imagenet...")
    logger.info("Starting ImageNet training process...")
    
    # Set environment variables for optimal performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU for Kaggle

    print("creating trainer instance...")
    # Create trainer instance
    trainer = ImageNetTrainer(config)

    print("Setting up trainer...")
    # Setup training
    trainer.setup()
    print("training start...")
    # Start training
    trainer.train()
    
    logger.info("Training completed!") 