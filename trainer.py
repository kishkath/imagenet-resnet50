import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist

from config import Config
from dataset import ImageNetDataset
from model import ResNet50Module

class ImageNetTrainer:
    def __init__(self, config: Config):
        self.config = config
        
    def setup(self):
        # Initialize dataset
        dataset = ImageNetDataset(self.config)
        self.train_loader, self.val_loader = dataset.get_dataloaders(
            distributed=self.config.distributed
        )
        
        # Initialize model
        self.model = ResNet50Module(config=self.config)
        
        # Enable gradient checkpointing in the model
        if hasattr(self.model.model, 'set_grad_checkpointing'):
            self.model.model.set_grad_checkpointing(enable=True)
        else:
            # Alternative way to enable gradient checkpointing for ResNet
            for module in self.model.model.modules():
                if isinstance(module, torch.nn.modules.container.Sequential):
                    module.gradient_checkpointing = True
        
        # Setup logging
        if self.config.use_wandb:
            self.logger = WandbLogger(
                project=self.config.project_name,
                log_model=True,
                group=f"resnet50_{self.config.batch_size}_{self.config.learning_rate}"
            )
        else:
            self.logger = True
        
        # Setup callbacks
        self.callbacks = [
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
        
    def train(self):
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
            log_every_n_steps=self.config.log_every_n_steps,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            deterministic=False,  # Faster training
            benchmark=True,  # Optimize CUDA kernels
            enable_progress_bar=True,
            enable_model_summary=True,
            detect_anomaly=False,  # Faster training
        )
        
        # Train
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
def train_imagenet(config: Config):
    # Set environment variables for optimal performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU for Kaggle
    
    # Create trainer instance
    trainer = ImageNetTrainer(config)
    
    # Setup training
    trainer.setup()
    
    # Start training
    trainer.train() 