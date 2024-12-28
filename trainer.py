import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GradientCheckpointingCallback
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist
from pytorch_lightning.plugins import MixedPrecisionPlugin

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
        self.model = ResNet50Module(self.config)
        
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
            LearningRateMonitor(logging_interval='step'),
            GradientCheckpointingCallback(use_gradient_checkpointing=True)
        ]
        
    def train(self):
        # Setup DDP strategy with NCCL backend
        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
        
        # Setup mixed precision plugin
        precision_plugin = MixedPrecisionPlugin(
            precision='16-mixed',
            device='cuda',
            scaler=None  # Let Lightning handle the scaler
        )
        
        # Initialize trainer with optimized settings
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu',
            devices=-1,  # Use all available GPUs
            strategy=strategy if self.config.distributed else None,
            precision='16-mixed',  # Use mixed precision
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
            plugins=[precision_plugin],
            sync_batchnorm=True  # Important for multi-GPU training
        )
        
        # Train
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
def train_imagenet(config: Config):
    # Set environment variables for optimal multi-GPU performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 GPUs
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    
    # Create trainer instance
    trainer = ImageNetTrainer(config)
    
    # Setup training
    trainer.setup()
    
    # Start training
    trainer.train() 